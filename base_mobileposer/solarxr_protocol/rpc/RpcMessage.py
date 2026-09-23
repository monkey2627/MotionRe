from solarxr_protocol.rpc.AddUnknownDeviceRequest import AddUnknownDeviceRequestT
from solarxr_protocol.rpc.AssignTrackerRequest import AssignTrackerRequestT
from solarxr_protocol.rpc.AutoBoneApplyRequest import AutoBoneApplyRequestT
from solarxr_protocol.rpc.AutoBoneCancelRecordingRequest import AutoBoneCancelRecordingRequestT
from solarxr_protocol.rpc.AutoBoneEpochResponse import AutoBoneEpochResponseT
from solarxr_protocol.rpc.AutoBoneProcessRequest import AutoBoneProcessRequestT
from solarxr_protocol.rpc.AutoBoneProcessStatusResponse import AutoBoneProcessStatusResponseT
from solarxr_protocol.rpc.AutoBoneStopRecordingRequest import AutoBoneStopRecordingRequestT
from solarxr_protocol.rpc.CancelUserHeightCalibration import CancelUserHeightCalibrationT
from solarxr_protocol.rpc.ChangeKeybindRequest import ChangeKeybindRequestT
from solarxr_protocol.rpc.ChangeMagToggleRequest import ChangeMagToggleRequestT
from solarxr_protocol.rpc.ChangeSettingsRequest import ChangeSettingsRequestT
from solarxr_protocol.rpc.ChangeSkeletonConfigRequest import ChangeSkeletonConfigRequestT
from solarxr_protocol.rpc.ClearDriftCompensationRequest import ClearDriftCompensationRequestT
from solarxr_protocol.rpc.ClearMountingResetRequest import ClearMountingResetRequestT
from solarxr_protocol.rpc.CloseSerialRequest import CloseSerialRequestT
from solarxr_protocol.rpc.DetectStayAlignedRelaxedPoseRequest import DetectStayAlignedRelaxedPoseRequestT
from solarxr_protocol.rpc.EnableStayAlignedRequest import EnableStayAlignedRequestT
from solarxr_protocol.rpc.EnableSteamVRDriverRequest import EnableSteamVRDriverRequestT
from solarxr_protocol.rpc.FirmwareUpdateRequest import FirmwareUpdateRequestT
from solarxr_protocol.rpc.FirmwareUpdateStatusResponse import FirmwareUpdateStatusResponseT
from solarxr_protocol.rpc.FirmwareUpdateStopQueuesRequest import FirmwareUpdateStopQueuesRequestT
from solarxr_protocol.rpc.ForgetDeviceRequest import ForgetDeviceRequestT
from solarxr_protocol.rpc.HeartbeatRequest import HeartbeatRequestT
from solarxr_protocol.rpc.HeartbeatResponse import HeartbeatResponseT
from solarxr_protocol.rpc.HeightRequest import HeightRequestT
from solarxr_protocol.rpc.HeightResponse import HeightResponseT
from solarxr_protocol.rpc.IgnoreTrackingChecklistStepRequest import IgnoreTrackingChecklistStepRequestT
from solarxr_protocol.rpc.InstalledInfoRequest import InstalledInfoRequestT
from solarxr_protocol.rpc.InstalledInfoResponse import InstalledInfoResponseT
from solarxr_protocol.rpc.KeybindRequest import KeybindRequestT
from solarxr_protocol.rpc.KeybindResponse import KeybindResponseT
from solarxr_protocol.rpc.LegTweaksTmpChange import LegTweaksTmpChangeT
from solarxr_protocol.rpc.LegTweaksTmpClear import LegTweaksTmpClearT
from solarxr_protocol.rpc.MagToggleRequest import MagToggleRequestT
from solarxr_protocol.rpc.MagToggleResponse import MagToggleResponseT
from solarxr_protocol.rpc.NewSerialDeviceResponse import NewSerialDeviceResponseT
from solarxr_protocol.rpc.OpenSerialRequest import OpenSerialRequestT
from solarxr_protocol.rpc.OpenUriRequest import OpenUriRequestT
from solarxr_protocol.rpc.OpenUriResponse import OpenUriResponseT
from solarxr_protocol.rpc.OverlayDisplayModeChangeRequest import OverlayDisplayModeChangeRequestT
from solarxr_protocol.rpc.OverlayDisplayModeRequest import OverlayDisplayModeRequestT
from solarxr_protocol.rpc.OverlayDisplayModeResponse import OverlayDisplayModeResponseT
from solarxr_protocol.rpc.RecordBVHRequest import RecordBVHRequestT
from solarxr_protocol.rpc.RecordBVHStatusRequest import RecordBVHStatusRequestT
from solarxr_protocol.rpc.RecordBVHStatus import RecordBVHStatusT
from solarxr_protocol.rpc.ResetRequest import ResetRequestT
from solarxr_protocol.rpc.ResetResponse import ResetResponseT
from solarxr_protocol.rpc.ResetStayAlignedRelaxedPoseRequest import ResetStayAlignedRelaxedPoseRequestT
from solarxr_protocol.rpc.SaveFileNotification import SaveFileNotificationT
from solarxr_protocol.rpc.SerialDevicesRequest import SerialDevicesRequestT
from solarxr_protocol.rpc.SerialDevicesResponse import SerialDevicesResponseT
from solarxr_protocol.rpc.SerialTrackerCustomCommandRequest import SerialTrackerCustomCommandRequestT
from solarxr_protocol.rpc.SerialTrackerFactoryResetRequest import SerialTrackerFactoryResetRequestT
from solarxr_protocol.rpc.SerialTrackerGetInfoRequest import SerialTrackerGetInfoRequestT
from solarxr_protocol.rpc.SerialTrackerGetWifiScanRequest import SerialTrackerGetWifiScanRequestT
from solarxr_protocol.rpc.SerialTrackerRebootRequest import SerialTrackerRebootRequestT
from solarxr_protocol.rpc.SerialUpdateResponse import SerialUpdateResponseT
from solarxr_protocol.rpc.ServerInfosRequest import ServerInfosRequestT
from solarxr_protocol.rpc.ServerInfosResponse import ServerInfosResponseT
from solarxr_protocol.rpc.SetPauseTrackingRequest import SetPauseTrackingRequestT
from solarxr_protocol.rpc.SetWifiRequest import SetWifiRequestT
from solarxr_protocol.rpc.SettingsRequest import SettingsRequestT
from solarxr_protocol.rpc.SettingsResetRequest import SettingsResetRequestT
from solarxr_protocol.rpc.SettingsResponse import SettingsResponseT
from solarxr_protocol.rpc.SkeletonConfigRequest import SkeletonConfigRequestT
from solarxr_protocol.rpc.SkeletonConfigResponse import SkeletonConfigResponseT
from solarxr_protocol.rpc.SkeletonResetAllRequest import SkeletonResetAllRequestT
from solarxr_protocol.rpc.StartUserHeightCalibration import StartUserHeightCalibrationT
from solarxr_protocol.rpc.StartWifiProvisioningRequest import StartWifiProvisioningRequestT
from solarxr_protocol.rpc.StatusSystemFixed import StatusSystemFixedT
from solarxr_protocol.rpc.StatusSystemRequest import StatusSystemRequestT
from solarxr_protocol.rpc.StatusSystemResponse import StatusSystemResponseT
from solarxr_protocol.rpc.StatusSystemUpdate import StatusSystemUpdateT
from solarxr_protocol.rpc.StopWifiProvisioningRequest import StopWifiProvisioningRequestT
from solarxr_protocol.rpc.TapDetectionSetupNotification import TapDetectionSetupNotificationT
from solarxr_protocol.rpc.TrackingChecklistRequest import TrackingChecklistRequestT
from solarxr_protocol.rpc.TrackingChecklistResponse import TrackingChecklistResponseT
from solarxr_protocol.rpc.TrackingPauseStateRequest import TrackingPauseStateRequestT
from solarxr_protocol.rpc.TrackingPauseStateResponse import TrackingPauseStateResponseT
from solarxr_protocol.rpc.UnknownDeviceHandshakeNotification import UnknownDeviceHandshakeNotificationT
from solarxr_protocol.rpc.UserHeightRecordingStatusResponse import UserHeightRecordingStatusResponseT
from solarxr_protocol.rpc.VRCConfigSettingToggleMute import VRCConfigSettingToggleMuteT
from solarxr_protocol.rpc.VRCConfigStateChangeResponse import VRCConfigStateChangeResponseT
from solarxr_protocol.rpc.VRCConfigStateRequest import VRCConfigStateRequestT
from solarxr_protocol.rpc.WifiProvisioningStatusResponse import WifiProvisioningStatusResponseT
# automatically generated by the FlatBuffers compiler, do not modify

# namespace: rpc

class RpcMessage(object):
    NONE = 0
    HeartbeatRequest = 1
    HeartbeatResponse = 2
    ResetRequest = 3
    ResetResponse = 4
    AssignTrackerRequest = 5
    SettingsRequest = 6
    SettingsResponse = 7
    ChangeSettingsRequest = 8
    ClearDriftCompensationRequest = 9
    RecordBVHRequest = 10
    RecordBVHStatus = 11
    SkeletonConfigRequest = 12
    ChangeSkeletonConfigRequest = 13
    SkeletonResetAllRequest = 14
    SkeletonConfigResponse = 15
    OpenSerialRequest = 16
    CloseSerialRequest = 17
    SetWifiRequest = 18
    SerialUpdateResponse = 19
    AutoBoneProcessRequest = 20
    AutoBoneProcessStatusResponse = 21
    AutoBoneEpochResponse = 22
    OverlayDisplayModeRequest = 23
    OverlayDisplayModeChangeRequest = 24
    OverlayDisplayModeResponse = 25
    SerialTrackerRebootRequest = 26
    SerialTrackerGetInfoRequest = 27
    SerialTrackerFactoryResetRequest = 28
    SerialDevicesRequest = 29
    SerialDevicesResponse = 30
    NewSerialDeviceResponse = 31
    StartWifiProvisioningRequest = 32
    StopWifiProvisioningRequest = 33
    WifiProvisioningStatusResponse = 34
    ServerInfosRequest = 35
    ServerInfosResponse = 36
    LegTweaksTmpChange = 37
    LegTweaksTmpClear = 38
    TapDetectionSetupNotification = 39
    SetPauseTrackingRequest = 40
    StatusSystemRequest = 41
    StatusSystemResponse = 42
    StatusSystemUpdate = 43
    StatusSystemFixed = 44
    ClearMountingResetRequest = 45
    HeightRequest = 46
    HeightResponse = 47
    AutoBoneApplyRequest = 48
    AutoBoneStopRecordingRequest = 49
    AutoBoneCancelRecordingRequest = 50
    SaveFileNotification = 51
    TrackingPauseStateRequest = 52
    TrackingPauseStateResponse = 53
    SerialTrackerGetWifiScanRequest = 54
    UnknownDeviceHandshakeNotification = 55
    AddUnknownDeviceRequest = 56
    ForgetDeviceRequest = 57
    FirmwareUpdateRequest = 58
    FirmwareUpdateStatusResponse = 59
    FirmwareUpdateStopQueuesRequest = 60
    SettingsResetRequest = 61
    MagToggleRequest = 62
    MagToggleResponse = 63
    ChangeMagToggleRequest = 64
    RecordBVHStatusRequest = 65
    VRCConfigStateRequest = 66
    VRCConfigStateChangeResponse = 67
    EnableStayAlignedRequest = 68
    DetectStayAlignedRelaxedPoseRequest = 69
    ResetStayAlignedRelaxedPoseRequest = 70
    SerialTrackerCustomCommandRequest = 71
    VRCConfigSettingToggleMute = 72
    TrackingChecklistRequest = 73
    TrackingChecklistResponse = 74
    IgnoreTrackingChecklistStepRequest = 75
    StartUserHeightCalibration = 76
    CancelUserHeightCalibration = 77
    UserHeightRecordingStatusResponse = 78
    KeybindRequest = 79
    ChangeKeybindRequest = 80
    KeybindResponse = 81
    InstalledInfoRequest = 82
    InstalledInfoResponse = 83
    OpenUriRequest = 84
    OpenUriResponse = 85
    EnableSteamVRDriverRequest = 86

def RpcMessageCreator(unionType, table):
    from flatbuffers.table import Table
    if not isinstance(table, Table):
        return None
    if unionType == RpcMessage.HeartbeatRequest:
        return HeartbeatRequestT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.HeartbeatResponse:
        return HeartbeatResponseT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.ResetRequest:
        return ResetRequestT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.ResetResponse:
        return ResetResponseT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.AssignTrackerRequest:
        return AssignTrackerRequestT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.SettingsRequest:
        return SettingsRequestT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.SettingsResponse:
        return SettingsResponseT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.ChangeSettingsRequest:
        return ChangeSettingsRequestT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.ClearDriftCompensationRequest:
        return ClearDriftCompensationRequestT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.RecordBVHRequest:
        return RecordBVHRequestT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.RecordBVHStatus:
        return RecordBVHStatusT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.SkeletonConfigRequest:
        return SkeletonConfigRequestT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.ChangeSkeletonConfigRequest:
        return ChangeSkeletonConfigRequestT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.SkeletonResetAllRequest:
        return SkeletonResetAllRequestT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.SkeletonConfigResponse:
        return SkeletonConfigResponseT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.OpenSerialRequest:
        return OpenSerialRequestT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.CloseSerialRequest:
        return CloseSerialRequestT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.SetWifiRequest:
        return SetWifiRequestT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.SerialUpdateResponse:
        return SerialUpdateResponseT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.AutoBoneProcessRequest:
        return AutoBoneProcessRequestT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.AutoBoneProcessStatusResponse:
        return AutoBoneProcessStatusResponseT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.AutoBoneEpochResponse:
        return AutoBoneEpochResponseT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.OverlayDisplayModeRequest:
        return OverlayDisplayModeRequestT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.OverlayDisplayModeChangeRequest:
        return OverlayDisplayModeChangeRequestT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.OverlayDisplayModeResponse:
        return OverlayDisplayModeResponseT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.SerialTrackerRebootRequest:
        return SerialTrackerRebootRequestT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.SerialTrackerGetInfoRequest:
        return SerialTrackerGetInfoRequestT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.SerialTrackerFactoryResetRequest:
        return SerialTrackerFactoryResetRequestT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.SerialDevicesRequest:
        return SerialDevicesRequestT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.SerialDevicesResponse:
        return SerialDevicesResponseT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.NewSerialDeviceResponse:
        return NewSerialDeviceResponseT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.StartWifiProvisioningRequest:
        return StartWifiProvisioningRequestT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.StopWifiProvisioningRequest:
        return StopWifiProvisioningRequestT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.WifiProvisioningStatusResponse:
        return WifiProvisioningStatusResponseT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.ServerInfosRequest:
        return ServerInfosRequestT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.ServerInfosResponse:
        return ServerInfosResponseT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.LegTweaksTmpChange:
        return LegTweaksTmpChangeT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.LegTweaksTmpClear:
        return LegTweaksTmpClearT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.TapDetectionSetupNotification:
        return TapDetectionSetupNotificationT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.SetPauseTrackingRequest:
        return SetPauseTrackingRequestT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.StatusSystemRequest:
        return StatusSystemRequestT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.StatusSystemResponse:
        return StatusSystemResponseT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.StatusSystemUpdate:
        return StatusSystemUpdateT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.StatusSystemFixed:
        return StatusSystemFixedT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.ClearMountingResetRequest:
        return ClearMountingResetRequestT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.HeightRequest:
        return HeightRequestT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.HeightResponse:
        return HeightResponseT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.AutoBoneApplyRequest:
        return AutoBoneApplyRequestT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.AutoBoneStopRecordingRequest:
        return AutoBoneStopRecordingRequestT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.AutoBoneCancelRecordingRequest:
        return AutoBoneCancelRecordingRequestT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.SaveFileNotification:
        return SaveFileNotificationT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.TrackingPauseStateRequest:
        return TrackingPauseStateRequestT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.TrackingPauseStateResponse:
        return TrackingPauseStateResponseT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.SerialTrackerGetWifiScanRequest:
        return SerialTrackerGetWifiScanRequestT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.UnknownDeviceHandshakeNotification:
        return UnknownDeviceHandshakeNotificationT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.AddUnknownDeviceRequest:
        return AddUnknownDeviceRequestT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.ForgetDeviceRequest:
        return ForgetDeviceRequestT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.FirmwareUpdateRequest:
        return FirmwareUpdateRequestT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.FirmwareUpdateStatusResponse:
        return FirmwareUpdateStatusResponseT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.FirmwareUpdateStopQueuesRequest:
        return FirmwareUpdateStopQueuesRequestT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.SettingsResetRequest:
        return SettingsResetRequestT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.MagToggleRequest:
        return MagToggleRequestT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.MagToggleResponse:
        return MagToggleResponseT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.ChangeMagToggleRequest:
        return ChangeMagToggleRequestT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.RecordBVHStatusRequest:
        return RecordBVHStatusRequestT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.VRCConfigStateRequest:
        return VRCConfigStateRequestT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.VRCConfigStateChangeResponse:
        return VRCConfigStateChangeResponseT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.EnableStayAlignedRequest:
        return EnableStayAlignedRequestT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.DetectStayAlignedRelaxedPoseRequest:
        return DetectStayAlignedRelaxedPoseRequestT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.ResetStayAlignedRelaxedPoseRequest:
        return ResetStayAlignedRelaxedPoseRequestT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.SerialTrackerCustomCommandRequest:
        return SerialTrackerCustomCommandRequestT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.VRCConfigSettingToggleMute:
        return VRCConfigSettingToggleMuteT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.TrackingChecklistRequest:
        return TrackingChecklistRequestT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.TrackingChecklistResponse:
        return TrackingChecklistResponseT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.IgnoreTrackingChecklistStepRequest:
        return IgnoreTrackingChecklistStepRequestT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.StartUserHeightCalibration:
        return StartUserHeightCalibrationT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.CancelUserHeightCalibration:
        return CancelUserHeightCalibrationT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.UserHeightRecordingStatusResponse:
        return UserHeightRecordingStatusResponseT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.KeybindRequest:
        return KeybindRequestT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.ChangeKeybindRequest:
        return ChangeKeybindRequestT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.KeybindResponse:
        return KeybindResponseT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.InstalledInfoRequest:
        return InstalledInfoRequestT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.InstalledInfoResponse:
        return InstalledInfoResponseT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.OpenUriRequest:
        return OpenUriRequestT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.OpenUriResponse:
        return OpenUriResponseT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == RpcMessage.EnableSteamVRDriverRequest:
        return EnableSteamVRDriverRequestT.InitFromBuf(table.Bytes, table.Pos)
    return None
