#!/usr/bin/env bash
set -euo pipefail

engine_directory=/home/boni/.cache/minestudio/engine/build/libs
source_jar="$engine_directory/mcprec-6.13.jar"
backup_jar="$engine_directory/mcprec-6.13-x86_64-original.jar"
arm64_jar="$engine_directory/mcprec-6.13-arm64.jar"
download_directory=/home/boni/.cache/minestudio/lwjgl-arm64-3.2.3
staging_directory="$download_directory/staging"
java_staging_directory="$download_directory/java-staging"
base_url=https://repo1.maven.org/maven2/org/lwjgl
lwjgl_version=3.2.3
modules=(lwjgl lwjgl-glfw lwjgl-jemalloc lwjgl-openal lwjgl-opengl lwjgl-stb lwjgl-tinyfd)

mkdir -p "$download_directory" "$staging_directory" "$java_staging_directory"
if [ ! -f "$backup_jar" ]; then
    cp "$source_jar" "$backup_jar"
fi
cp "$backup_jar" "$arm64_jar"
rm -f "$staging_directory"/*.so "$staging_directory"/*.so.git "$staging_directory"/*.so.sha1
rm -rf "$java_staging_directory"/org

for module in "${modules[@]}"; do
    native_jar="$download_directory/$module-$lwjgl_version-natives-linux-arm64.jar"
    native_url="$base_url/$module/$lwjgl_version/$module-$lwjgl_version-natives-linux-arm64.jar"
    java_jar="$download_directory/$module-$lwjgl_version.jar"
    java_url="$base_url/$module/$lwjgl_version/$module-$lwjgl_version.jar"
    if [ ! -f "$native_jar" ]; then
        curl --fail --location --retry 3 --output "$native_jar" "$native_url"
    fi
    if [ ! -f "$java_jar" ]; then
        curl --fail --location --retry 3 --output "$java_jar" "$java_url"
    fi
    unzip -jo "$native_jar" '*.so' -d "$staging_directory" >/dev/null
    unzip -oq "$java_jar" 'org/*' -d "$java_staging_directory"
done

for native_library in "$staging_directory"/*.so; do
    sha1sum "$native_library" | cut -d ' ' -f 1 > "$native_library.sha1"
done

(
    cd "$staging_directory"
    zip -q -u "$arm64_jar" ./*.so ./*.so.sha1
)
(
    cd "$java_staging_directory"
    zip -qr -u "$arm64_jar" ./org
)

for native_library in "$staging_directory"/*.so; do
    file "$native_library"
done
sha256sum "$backup_jar" "$arm64_jar"
