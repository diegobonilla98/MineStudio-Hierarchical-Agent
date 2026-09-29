$distribution = "Ubuntu-24.04"
$projectDirectory = "/mnt/d/HugeProjects/MineStudio"
$python = "/home/diego/.local/bin/micromamba run -n minestudio python"
$validator = "run/validate_coach_dataset.py"
$untranslatablePath = "E:\Tools\COLMAP\4.0.4-cuda\bin"
$originalPath = $env:PATH
$env:PATH = (($originalPath -split ";") | Where-Object { $_ -ne $untranslatablePath }) -join ";"

try {
    & "$env:WINDIR\System32\wsl.exe" -d $distribution -- bash -lc "cd '$projectDirectory' && $python '$validator'"
    $validationExitCode = $LASTEXITCODE
} finally {
    $env:PATH = $originalPath
}

exit $validationExitCode
