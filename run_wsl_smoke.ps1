$distribution = "Ubuntu-24.04"
$rootPrefix = "/home/diego/.local/share/mamba"
$micromamba = "/home/diego/.local/bin/micromamba"
$environmentName = "minestudio"
$mineStudioDirectory = "/home/diego/.cache/minestudio"
$untranslatablePath = "E:\Tools\COLMAP\4.0.4-cuda\bin"
$originalPath = $env:PATH
$env:PATH = (($originalPath -split ";") | Where-Object { $_ -ne $untranslatablePath }) -join ";"
$linuxCommand = @"
export MINESTUDIO_DIR='$mineStudioDirectory'
MAMBA_ROOT_PREFIX='$rootPrefix' '$micromamba' run -n '$environmentName' python -m minestudio.simulator.entry -y
"@

try {
    wsl.exe -d $distribution -- bash -lc $linuxCommand
} finally {
    $env:PATH = $originalPath
}
