<#
.SYNOPSIS
    Prepare this folder as a Git repository and push it to GitHub.

.DESCRIPTION
    Step 1  checks that git is installed.
    Step 2  replaces the OWNER / <your-user> placeholders in README.md and
            pyproject.toml with your GitHub user name (or organisation).
    Step 3  initialises the repository, makes the first commit and, with
            -Push, adds the remote and pushes.

    Nothing is pushed unless -Push is given.

.PARAMETER UserName
    Your GitHub user name or organisation, e.g. "shijian-wang".

.PARAMETER RepoName
    Repository name on GitHub. Defaults to "parareal-spectral-sl".

.PARAMETER Push
    Add the remote and push. The repository must already exist on GitHub.

.EXAMPLE
    .\publish_to_github.ps1 -UserName shijian-wang
    Prepares and commits, prints the remaining two commands.

.EXAMPLE
    .\publish_to_github.ps1 -UserName shijian-wang -Push
    Also creates the remote and pushes to origin/main.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$UserName,
    [string]$RepoName = "parareal-spectral-sl",
    [switch]$Push
)

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

# GitHub user names are ASCII: letters, digits and hyphens. Rejecting anything
# else stops a stray placeholder (or a non-ASCII copy-paste) from being written
# into README.md and pyproject.toml.
if ($UserName -notmatch '^[A-Za-z0-9]([A-Za-z0-9-]{0,38})$') {
    Write-Host "UserName '$UserName' is not a valid GitHub user name." -ForegroundColor Red
    Write-Host "Expected ASCII letters, digits or hyphens, e.g. -UserName Shijian66" -ForegroundColor Red
    exit 2
}

Write-Host "=== 1. git ===" -ForegroundColor Cyan
$git = Get-Command git -ErrorAction SilentlyContinue
if (-not $git) {
    Write-Host "git not found. Install it first:" -ForegroundColor Red
    Write-Host "    winget install --id Git.Git -e --source winget"
    Write-Host "then open a new terminal (so PATH is refreshed) and run this script again."
    exit 1
}
Write-Host ("git {0}" -f ((git --version) -replace 'git version ', ''))

$name = git config --global user.name
$email = git config --global user.email
if (-not $name -or -not $email) {
    Write-Host "Set your identity once (used in every commit):" -ForegroundColor Yellow
    Write-Host '    git config --global user.name  "Shijian Wang"'
    Write-Host '    git config --global user.email "you@example.com"'
    exit 1
}
Write-Host ("committing as {0} <{1}>" -f $name, $email)

Write-Host ""
Write-Host "=== 2. placeholders ===" -ForegroundColor Cyan
$url = "https://github.com/$UserName/$RepoName"
foreach ($file in @("README.md", "pyproject.toml")) {
    $path = Join-Path $PSScriptRoot $file
    if (-not (Test-Path $path)) { continue }
    $text = Get-Content -LiteralPath $path -Raw -Encoding UTF8
    $before = ([regex]::Matches($text, "OWNER|<your-user>")).Count
    $text = $text -replace "OWNER", $UserName
    $text = $text -replace "<your-user>/parareal-spectral-sl", "$UserName/$RepoName"
    $text = $text -replace "<your-user>", $UserName
    [System.IO.File]::WriteAllText($path, $text, [System.Text.UTF8Encoding]::new($false))
    Write-Host ("  {0}: replaced {1} placeholder(s)" -f $file, $before) -ForegroundColor Green
}

Write-Host ""
Write-Host "=== 3. repository ===" -ForegroundColor Cyan
if (-not (Test-Path (Join-Path $PSScriptRoot ".git"))) {
    git init -b main | Out-Null
    Write-Host "  initialised (branch main)"
} else {
    Write-Host "  already a git repository"
}

git add -A
$staged = (git diff --cached --name-only | Measure-Object).Count
Write-Host ("  staged {0} file(s)" -f $staged)
if ($staged -eq 0) {
    Write-Host "  nothing to commit" -ForegroundColor Yellow
} else {
    git commit -m "Spectral semi-Lagrangian Parareal propagator: code, tests and figures" 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) { Write-Host "  committed" -ForegroundColor Green }
    else { Write-Host "  commit failed (see the message above)" -ForegroundColor Red; exit 1 }
}

Write-Host ""
Write-Host "=== 4. push ===" -ForegroundColor Cyan
$hasRemote = (git remote) -contains "origin"
if ($Push) {
    if (-not $hasRemote) { git remote add origin "$url.git" }
    git branch -M main
    git push -u origin main
    if ($LASTEXITCODE -eq 0) { Write-Host "  pushed to $url" -ForegroundColor Green }
    else {
        Write-Host "  push failed. Try SSH over port 443 (see README) or a proxy:" -ForegroundColor Red
        Write-Host "    git config --global http.proxy http://127.0.0.1:7890"
        exit 1
    }
} else {
    Write-Host "Not pushed (-Push not given). Do it in two commands:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host ("  1) create the empty repository at {0}/new" -f "https://github.com/$UserName")
    Write-Host "  2) then run:"
    Write-Host ""
    Write-Host ("     git remote add origin {0}.git" -f $url)
    Write-Host "     git branch -M main"
    Write-Host "     git push -u origin main"
}
Write-Host ""
