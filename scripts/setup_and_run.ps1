# path: scripts/setup_and_run.ps1
# PowerShell 5/7 – Windows
param(
  [Parameter(Position=0,Mandatory=$true)]
  [ValidateSet("setup","cache-models","run-img","run-dir","run-vid","run-cam","docker-build","docker-run","help")]
  [string]$cmd,
  [Parameter(Position=1)] [string]$arg1,
  [switch]$SkipTests
)
$ErrorActionPreference = "Stop"
$AppEntry = "tools/describe_objects.py"
$ReqFile = "requirements.txt"
$VenvDir = ".venv"

function Usage {
@"
Uso:
  scripts\setup_and_run.ps1 setup [-SkipTests]
  scripts\setup_and_run.ps1 cache-models [-Lite]
  scripts\setup_and_run.ps1 run-img <imagen> [-Out out] [-Lite] [-Cpu] [-Score 0.5] [-MaxSide 1280] [-Lang es]
  scripts\setup_and_run.ps1 run-dir <carpeta> [-Out out] [-Lite] [-Cpu] [-Score 0.5] [-MaxSide 1280] [-Lang es]
  scripts\setup_and_run.ps1 run-vid <video> [-Out out] [-Lite] [-Cpu] [-Fps 1.0] [-Score 0.5] [-MaxSide 1280] [-Lang es]
  scripts\setup_and_run.ps1 run-cam [-Out out] [-Lite] [-Cpu] [-Fps 1.0] [-MaxSide 1280] [-Lang es]
  scripts\setup_and_run.ps1 docker-build [-Tag objdesc:cpu]
  scripts\setup_and_run.ps1 docker-run <run-img|run-dir|run-vid> <ruta> [-Out out] [-Lite] [-Cpu] [-Fps 1.0] [-Score 0.5] [-MaxSide 1280] [-Tag objdesc:cpu] [-Lang es]
"@
}

function Assert-File($p){ if(!(Test-Path $p)){ throw "Falta archivo: $p" } }
function Activate-Venv {
  if(!(Test-Path $VenvDir)){ py -3 -m venv $VenvDir }
  & "$VenvDir\Scripts\Activate.ps1" | Out-Null
  python --version | Write-Host
}
function Install-Req {
  Assert-File $ReqFile
  $env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
  try { python -m pip --version | Out-Null } catch { python -m ensurepip --upgrade --default-pip } # evita romper pip
  python -m pip install --upgrade --no-cache-dir setuptools wheel
  python -m pip install --no-cache-dir -r $ReqFile
}
function Cache-Models([switch]$Lite){
  $py = @"
from transformers import pipeline
lite = ${$Lite.IsPresent}
det_model = 'hustvl/yolos-tiny' if lite else 'facebook/detr-resnet-50'
cap_model = 'Salesforce/blip-image-captioning-base' if lite else 'Salesforce/blip-image-captioning-large'
pipeline('object-detection', model=det_model, device=-1)
pipeline('image-to-text', model=cap_model, device=-1)
print('cached', det_model, cap_model)
"@
  $tmp = [System.IO.Path]::GetTempFileName()
  Set-Content -Path $tmp -Value $py -Encoding UTF8
  python $tmp
  Remove-Item $tmp -Force
}
function Parse-Common([hashtable]$defaults){
  $opts = $defaults.Clone(); $i = 0
  while($i -lt $args.Count){
    switch -regex ($args[$i]) {
      '^-Out$'       { $opts.Out = $args[$i+1]; $i+=2; continue }
      '^-Lite$'      { $opts.Lite = $true;      $i+=1; continue }
      '^-Cpu$'       { $opts.Cpu = $true;       $i+=1; continue }
      '^-Score$'     { $opts.Score = [double]$args[$i+1]; $i+=2; continue }
      '^-MaxSide$'   { $opts.MaxSide = [int]$args[$i+1];  $i+=2; continue }
      '^-Fps$'       { $opts.Fps = [double]$args[$i+1];   $i+=2; continue }
      '^-Tag$'       { $opts.Tag = $args[$i+1]; $i+=2; continue }
      '^-Lang$'      { $opts.Lang = $args[$i+1]; $i+=2; continue }
      default        { throw "Flag desconocida: $($args[$i])" }
    }
  }
  return $opts
}

switch ($cmd) {
  "help" { Usage; break }

  "setup" {
    $skip = $SkipTests.IsPresent -or $args -contains "-SkipTests"
    Activate-Venv
    Install-Req
    if((Test-Path tests) -and -not $skip){
      try { pytest -q } catch { throw "Tests fallaron" }
    }
  }

  "cache-models" {
    $opts = Parse-Common(@{ Lite=$false })
    Activate-Venv; Install-Req; Cache-Models -Lite:$opts.Lite
  }

  "run-img" {
    if([string]::IsNullOrWhiteSpace($arg1)){ throw "Falta <imagen>" }
    $opts = Parse-Common(@{ Out="out"; Lite=$false; Cpu=$false; Score=0.5; MaxSide=0; Lang="en" })
    Assert-File $AppEntry; Activate-Venv
    $flags = @("--out",$opts.Out,"--score",$opts.Score)
    if($opts.Lite){ $flags += "--lite" }
    if($opts.Cpu){  $flags += "--cpu" }
    if($opts.MaxSide -gt 0){ $flags += @("--max-side",$opts.MaxSide) }
    if($opts.Lang){ $flags += @("--lang",$opts.Lang) }
    python $AppEntry --input $arg1 @flags
  }

  "run-dir" {
    if([string]::IsNullOrWhiteSpace($arg1)){ throw "Falta <carpeta>" }
    $opts = Parse-Common(@{ Out="out"; Lite=$false; Cpu=$false; Score=0.5; MaxSide=0; Lang="en" })
    Assert-File $AppEntry; Activate-Venv
    $flags = @("--out",$opts.Out,"--score",$opts.Score)
    if($opts.Lite){ $flags += "--lite" }
    if($opts.Cpu){  $flags += "--cpu" }
    if($opts.MaxSide -gt 0){ $flags += @("--max-side",$opts.MaxSide) }
    if($opts.Lang){ $flags += @("--lang",$opts.Lang) }
    python $AppEntry --input $arg1 @flags
  }

  "run-vid" {
    if([string]::IsNullOrWhiteSpace($arg1)){ throw "Falta <video>" }
    $opts = Parse-Common(@{ Out="out"; Lite=$false; Cpu=$false; Score=0.5; MaxSide=0; Fps=1.0; Lang="en" })
    Assert-File $AppEntry; Activate-Venv
    $flags = @("--out",$opts.Out,"--score",$opts.Score,"--fps",$opts.Fps)
    if($opts.Lite){ $flags += "--lite" }
    if($opts.Cpu){  $flags += "--cpu" }
    if($opts.MaxSide -gt 0){ $flags += @("--max-side",$opts.MaxSide) }
    if($opts.Lang){ $flags += @("--lang",$opts.Lang) }
    python $AppEntry --video $arg1 @flags
  }

  "run-cam" {
    $opts = Parse-Common(@{ Out="out"; Lite=$false; Cpu=$false; MaxSide=0; Fps=1.0; Lang="en" })
    Assert-File $AppEntry; Activate-Venv
    $flags = @("--out",$opts.Out,"--fps",$opts.Fps)
    if($opts.Lite){ $flags += "--lite" }
    if($opts.Cpu){  $flags += "--cpu" }
    if($opts.MaxSide -gt 0){ $flags += @("--max-side",$opts.MaxSide) }
    if($opts.Lang){ $flags += @("--lang",$opts.Lang) }
    python $AppEntry --webcam @flags
  }

  "docker-build" {
    $opts = Parse-Common(@{ Tag="objdesc:cpu" })
    Assert-File "Dockerfile"
    docker build -t $opts.Tag .
  }

  "docker-run" {
    if([string]::IsNullOrWhiteSpace($arg1)){ throw "Falta subcomando: run-img|run-dir|run-vid" }
    $sub = $arg1
    $route = $args | Select-Object -First 1
    if([string]::IsNullOrWhiteSpace($route)){ throw "Falta <ruta>" }
    $opts = Parse-Common(@{ Out="out"; Lite=$false; Cpu=$true; Score=0.5; MaxSide=0; Fps=1.0; Tag="objdesc:cpu"; Lang="en" })
    $vol = "$($PWD.Path):/app"
    $common = @("--out","/app/$($opts.Out)","--score",$opts.Score)
    if($opts.Lite){ $common += "--lite" }
    if($opts.Cpu){  $common += "--cpu" }
    if($opts.MaxSide -gt 0){ $common += @("--max-side",$opts.MaxSide) }
    if($opts.Lang){ $common += @("--lang",$opts.Lang) }

    switch ($sub) {
      "run-img" { docker run --rm -v $vol $opts.Tag --input "/app/$route" @common }
      "run-dir" { docker run --rm -v $vol $opts.Tag --input "/app/$route" @common }
      "run-vid" { docker run --rm -v $vol $opts.Tag --video "/app/$route" @($common + @("--fps",$opts.Fps)) }
      default   { throw "Subcomando inválido: $sub (usa run-img|run-dir|run-vid)" }
    }
  }

  default { Usage }
}
