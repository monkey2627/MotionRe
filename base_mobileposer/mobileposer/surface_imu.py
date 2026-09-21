"""Fixed SMPL surface attachments, calibrated IMU synthesis and placement preview."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from mobileposer.articulate.model import ParametricModel
from mobileposer.articulate import math
from mobileposer.config import paths

DEFAULT_CONFIG = Path(__file__).parent / 'configs/no_head_surface_imu.json'


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_attachments(path=DEFAULT_CONFIG, model=None):
    config = json.loads(Path(path).read_text())
    if config['smpl_sha256'] != file_hash(paths.smpl_file):
        raise ValueError('Attachment configuration does not match the SMPL model')
    model = model or ParametricModel(paths.smpl_file)
    required = {'lw':18,'rw':19,'lu':16,'ru':17,'lt':1,'rt':2,'ls':4,'rs':5,'lf':7,'rf':8,'waist':3}
    for name, joint in required.items():
        a = config['attachments'][name]
        if a['joint'] != joint:
            raise ValueError(f'{name}: unexpected bone mapping')
        if not 0 <= a['face'] < len(model.face):
            raise ValueError(f'{name}: invalid face')
        if list(map(int, model.face[a['face']])) != a['vertices']:
            raise ValueError(f'{name}: face/vertex mismatch')
        w = np.asarray(a['weights'])
        r = np.asarray(a['mount_rotation'])
        if w.shape != (3,) or not np.isfinite(w).all() or (w < 0).any() or not np.isclose(w.sum(), 1):
            raise ValueError(f'{name}: invalid barycentric weights')
        if r.shape != (3,3) or not np.isfinite(r).all() or not np.allclose(r.T @ r,np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(r),1,atol=1e-5):
            raise ValueError(f'{name}: invalid mounting rotation')
    return config


def attachment_points(vertices, config, labels):
    ids = torch.tensor([config['attachments'][x]['vertices'] for x in labels], device=vertices.device)
    weights = vertices.new_tensor([config['attachments'][x]['weights'] for x in labels])
    return (vertices[:, ids] * weights[None, :, :, None]).sum(2)


def surface_kinematics(model, pose, shape, tran, labels, config, chunk_size=128):
    """Keep selected trajectories, not full sequence meshes; differentiate after concatenation."""
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    rotations, joints, points = [], [], []
    bone_ids = [config['attachments'][x]['joint'] for x in labels]
    mounts = pose.new_tensor([config['attachments'][x]['mount_rotation'] for x in labels])
    with torch.no_grad():
        for start in range(0, len(pose), chunk_size):
            r, j, v = model.forward_kinematics(pose[start:start+chunk_size], shape, tran[start:start+chunk_size], calc_mesh=True)
            device_r = r[:, bone_ids] @ mounts
            calibrated_r = device_r @ mounts.transpose(-1,-2)
            rotations.append(torch.cat((calibrated_r, r[:, :1]), dim=1))
            joints.append(j)
            points.append(torch.cat((attachment_points(v, config, labels), j[:, :1]), dim=1))
    return torch.cat(rotations), torch.cat(joints), torch.cat(points)


def make_config():
    """Intersect anatomical rays with the actual neutral SMPL mesh, then freeze barycentrics."""
    model = ParametricModel(paths.smpl_file)
    joint, vertex = model.get_zero_pose_joint_and_vertex(torch.zeros(10))
    j, v = joint[0].numpy(), vertex[0].numpy()
    specifications = {}
    for side, offset, sign in [('l',0,1),('r',1,-1)]:
        for label, bone, child, fraction, direction, description in [
            ('w',18,20,.75,[0,1,0],'distal forearm lateral surface; image forearm tracker'),
            ('u',16,18,.82,[0,1,0],'lateral upper arm just above elbow'),
            ('t',1,4,.86,[0,0,1],'anterior distal thigh above knee'),
            ('s',4,7,.88,[sign,0,0],'lateral distal shank above ankle'),
            ('f',7,10,.60,[0,1,0],'dorsal foot; synthetic location, no assigned physical foot tracker'),
        ]:
            bone, child = bone+offset, child+offset
            specifications[side+label] = (bone, (1-fraction)*j[bone]+fraction*j[child], np.array(direction,dtype=float), description, fraction)
    specifications['waist'] = (3,np.array([0,j[3,1],j[3,2]]),np.array([0,0,1.]),'anterior waist midline at spine1 height',None)
    tri = v[model.face]
    e1, e2 = tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]
    attachments = {}
    for name,(bone,origin,direction,description,fraction) in specifications.items():
        h = np.cross(direction,e2)
        det = (e1*h).sum(1)
        safe = np.where(abs(det)>1e-10,det,1.)
        s = origin-tri[:,0]
        u = (s*h).sum(1)/safe
        q = np.cross(s,e1)
        b = (q*direction).sum(1)/safe
        t = (e2*q).sum(1)/safe
        eligible = (abs(det)>1e-10)&(u>=0)&(b>=0)&(u+b<=1)&(t>0)&(t<.25)
        candidates=np.flatnonzero(eligible)
        if not len(candidates):
            raise ValueError(f'No surface intersection for {name}')
        face = int(candidates[np.argmin(t[candidates])])
        normal = np.cross(e1[face],e2[face]); normal /= np.linalg.norm(normal)
        if normal @ direction < 0: normal = -normal
        tangent = np.array([0.,1.,0.])
        if abs(tangent @ normal)>.9: tangent=np.array([1.,0.,0.])
        tangent -= normal*(normal@tangent); tangent/=np.linalg.norm(tangent)
        mounting=np.stack((tangent,np.cross(normal,tangent),normal),axis=1)
        attachments[name] = dict(joint=bone,face=face,vertices=list(map(int,model.face[face])),weights=[float(1-u[face]-b[face]),float(u[face]),float(b[face])],mount_rotation=mounting.tolist(),description=description,bone_fraction=fraction,neutral_position_m=(origin+t[face]*direction).tolist())
    return dict(version=1,smpl_sha256=file_hash(paths.smpl_file),reference='SlimeVR screenshot; forearm (not hand), anatomical left/right',orientation='bone_calibrated',acceleration='world_linear_m_s2',pose_blendshape=False,attachments=attachments)


def render_preview(config, output, labels=None, layout_name=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams['font.family']='Noto Sans CJK JP'
    plt.rcParams['axes.unicode_minus']=False
    model=ParametricModel(paths.smpl_file)
    aa=torch.zeros(1,24,3)
    aa[0,16,2]=-1.15; aa[0,17,2]=1.15
    pose=math.axis_angle_to_rotation_matrix(aa).reshape(1,24,3,3)
    _,joint,vertex=model.forward_kinematics(pose,torch.zeros(10),torch.zeros(1,3),calc_mesh=True)
    labels=list(config['attachments']) if labels is None else list(labels)
    p=attachment_points(vertex,config,labels)[0].numpy()
    v=vertex[0].numpy(); j=joint[0].numpy()
    from matplotlib.collections import PolyCollection
    colors={'w':'#ed8431','u':'#ae52d4','t':'#168bdd','s':'#08a68d','f':'#d65189','waist':'#b58c00'}
    names={'w':'小臂','u':'上臂','t':'大腿','s':'小腿','f':'足背','waist':'前腰'}
    fig,axes=plt.subplots(1,3,figsize=(18,11),facecolor='#f6f8fb')
    for idx,(ax,title) in enumerate(zip(axes,['正面（人体左侧在图左）','人体左侧（前方在右）','背面（圆圈为透视标记）'])):
        horizontal = -v[:,0] if idx==0 else v[:,2] if idx==1 else v[:,0]
        depth = v[:,2] if idx==0 else v[:,0] if idx==1 else -v[:,2]
        xy=np.stack((horizontal,v[:,1]),axis=-1)
        polygons=xy[model.face]
        order=np.argsort(depth[model.face].mean(1))
        tri=v[model.face]; normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]); normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-9)
        light=np.array([.3,.5,1.]); light/=np.linalg.norm(light)
        shade=.65+.28*np.abs(normal@light)
        facecolors=np.stack((shade*.83,shade*.90,shade),axis=1)
        ax.add_collection(PolyCollection(polygons[order],facecolors=facecolors[order],edgecolors='none',rasterized=True))
        for k,name in enumerate(labels):
            if idx==1 and name.startswith('r'): continue
            key='waist' if name=='waist' else name[1]; color=colors[key]
            px=-p[k,0] if idx==0 else p[k,2] if idx==1 else p[k,0]
            py=p[k,1]
            left=(name.startswith('l') if idx==0 else name.startswith('r') if idx==2 else key in ('w','u','s'))
            tx=-.58 if left else .58
            ty=py
            if key=='s': ty+=.035
            if key=='f': ty-=.035
            if key=='waist': ty+=.055
            label=name+' '+('左' if name.startswith('l') else '右' if name.startswith('r') else '')+names[key]
            ax.plot([px,tx*.85],[py,ty],color=color,lw=1.2,zorder=5,linestyle='--' if idx==2 else '-')
            ax.scatter(px,py,s=65,facecolors='none' if idx==2 else color,edgecolors=color if idx==2 else 'white',linewidths=1.8,zorder=6)
            ax.text(tx,ty,label,color=color,fontsize=12,ha='right' if left else 'left',va='center',zorder=7)
        ax.set_xlim(-.88,.88); ax.set_ylim(-1.07,.87); ax.set_aspect('equal'); ax.axis('off'); ax.set_title(title,fontsize=16,pad=20)
    fig.suptitle('SMPL 皮肤表面 IMU 选点预览',fontsize=25,y=.96)
    if layout_name is None:
        subtitle = '六组布局共用 11 个安装点 · 彩点来自固定三角面及重心坐标 · wrists 对应小臂而非手部'
    else:
        subtitle = f'{layout_name} · 5 IMU\n模型输入顺序：' + ' → '.join(labels)
    fig.text(.5,.885 if layout_name else .905,subtitle,ha='center',fontsize=14,linespacing=1.6)
    descriptions = {'w':'小臂：肘→腕的 75% 处外侧', 'u':'上臂：肩→肘的 82% 处外侧',
                    't':'大腿：髋→膝的 86% 处前侧', 's':'小腿：膝→踝的 88% 处外侧',
                    'f':'脚：足背', 'waist':'腰：前腹中线、spine1 高度'}
    keys = list(dict.fromkeys('waist' if label == 'waist' else label[1] for label in labels))
    selected = [descriptions[key] for key in keys]
    description = '    '.join(selected[:3])
    if len(selected) > 3:
        description += '\n' + '    '.join(selected[3:])
    note = '背面圆圈是遮挡点的透视标记，不表示绑在背部；侧面只标注左侧和前腰。'
    if 'f' in keys:
        note += '\n足背为合成候选，原图脚部 tracker 未分配。'
    fig.text(.5,.055,description + '\n' + note + '\n使用项目训练用 SMPL；彩点为固定面片重心插值点，非截图拟合或实测标定。',ha='center',fontsize=12,linespacing=1.7)
    fig.subplots_adjust(left=.03,right=.97,top=.85,bottom=.19,wspace=.12)
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=170,bbox_inches='tight'); plt.close(fig)
    print(output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--initialize',action='store_true',help='Create frozen attachments; refuses overwriting existing configuration')
    parser.add_argument('--preview',type=Path,default=paths.root_dir/'results/surface_imu/placement_preview.png')
    parser.add_argument('--layouts', nargs='+', default=None, help='Render separate layout previews; use all for all six layouts')
    args=parser.parse_args()
    if args.initialize:
        config=make_config()
        args.config.parent.mkdir(parents=True,exist_ok=True)
        with args.config.open('x') as f: json.dump(config,f,indent=2)
    config = load_attachments(args.config)
    if args.layouts:
        from mobileposer.no_head_layouts import LAYOUTS
        layout_names = list(LAYOUTS) if args.layouts == ['all'] else args.layouts
        unknown = set(layout_names) - set(LAYOUTS)
        if unknown:
            parser.error(f'Unknown layouts: {sorted(unknown)}')
        for layout_name in layout_names:
            output = args.preview.parent / 'layouts' / f'{layout_name}.png'
            render_preview(config, output, LAYOUTS[layout_name]['labels'], layout_name)
    else:
        render_preview(config,args.preview)

if __name__=='__main__': main()
