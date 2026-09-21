import groovy.json.JsonSlurper

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "org.mobileposer.smpl"
    compileSdk = 35
    defaultConfig {
        applicationId = "org.mobileposer.smpl"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"
        ndk { abiFilters += listOf("arm64-v8a", "x86_64") }
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }
    sourceSets["main"].assets.srcDir(rootProject.file("assets_generated"))
    androidResources { noCompress += "onnx" }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    buildTypes {
        release { isMinifyEnabled = false }
    }
}

dependencies {
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.19.2")
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20240303")
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
}

tasks.register("checkModelAssets") {
    doLast {
        val dir = rootProject.file("assets_generated")
        check(dir.resolve("manifest.json").isFile) {
            "Run python -m mobileposer.mobile_export from the repository first."
        }
        check(dir.listFiles { f -> f.extension == "onnx" }?.size == 6) { "All six ONNX models are required." }
        val expected = setOf("wrists_thighs_waist", "wrists_shanks_waist", "wrists_feet_waist",
            "upperarms_thighs_waist", "wrists_upperarms_waist", "legs_waist")
        val manifest = JsonSlurper().parse(dir.resolve("manifest.json")) as Map<*, *>
        val models = manifest["models"] as List<*>
        check(models.size == 6 && models.map { (it as Map<*, *>)["layout"] }.toSet() == expected) {
            "Manifest must describe all six layouts. Re-run export without --layouts."
        }
        for (layout in expected) {
            for (suffix in listOf(".onnx", ".input.json", ".reference.json")) {
                check(dir.resolve(layout + suffix).isFile) { "Missing asset: $layout$suffix" }
            }
        }
    }
}
tasks.named("preBuild") { dependsOn("checkModelAssets") }
