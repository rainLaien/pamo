param(
    [switch]$Repair,
    [switch]$RemoveUnreachableLocalProxy,
    [switch]$Deep,
    [string]$ProxyHost = "127.0.0.1",
    [int[]]$CommonProxyPorts = @(7890, 7891, 7892, 7893, 1080, 10808)
)

$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"

# ============================================================
# CodexNetworkDoctor.ps1
#
# 用途：
#   诊断本地 Codex 出现：
#
#   Reconnecting... waiting for network
#   Connection failed: error sending request
#
# 检查范围：
#   - DNS
#   - TCP 443
#   - HTTPS / TLS
#   - Windows WinINET Proxy
#   - WinHTTP Proxy
#   - HTTP_PROXY / HTTPS_PROXY / ALL_PROXY
#   - ~/.codex/.env
#   - ~/.codex/config.toml
#   - 本地代理端口
#   - Codex / Node 进程
#   - Codex 本地文件异常
#
# 使用：
#   powershell -ExecutionPolicy Bypass -File .\CodexNetworkDoctor.ps1
#
# 安全修复：
#   powershell -ExecutionPolicy Bypass -File .\CodexNetworkDoctor.ps1 -Repair
#   Repair only flushes DNS. Persistent local proxy/endpoint removal additionally
#   requires -RemoveUnreachableLocalProxy and repeated failed loopback probes.
#
# 深度诊断：
#   powershell -ExecutionPolicy Bypass -File .\CodexNetworkDoctor.ps1 -Deep
# ============================================================


# ------------------------------------------------------------
# Global
# ------------------------------------------------------------

$script:Failures = @()
$script:Warnings = @()
$script:Passes   = @()

$UserHome = [Environment]::GetFolderPath("UserProfile")
$CodexHome = if ($env:CODEX_HOME) {
    $env:CODEX_HOME
}
else {
    Join-Path $UserHome ".codex"
}

$CodexEnvFile = Join-Path $CodexHome ".env"
$CodexConfigFile = Join-Path $CodexHome "config.toml"

$BackupRoot = Join-Path $UserHome (
    ".codex-network-doctor-backup-" + (Get-Date -Format "yyyyMMdd-HHmmss")
)

$Targets = @(
    "chatgpt.com",
    "api.openai.com",
    "openai.com"
)


# ------------------------------------------------------------
# Console helpers
# ------------------------------------------------------------

function Write-Title {
    param([string]$Text)

    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host " $Text" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
}

function Write-Section {
    param([string]$Text)

    Write-Host ""
    Write-Host "---- $Text ----" -ForegroundColor DarkCyan
}

function Write-Pass {
    param([string]$Text)

    $Text = Protect-DiagnosticText $Text
    Write-Host "[PASS] $Text" -ForegroundColor Green
    $script:Passes += $Text
}

function Write-Warn {
    param([string]$Text)

    $Text = Protect-DiagnosticText $Text
    Write-Host "[WARN] $Text" -ForegroundColor Yellow
    $script:Warnings += $Text
}

function Write-Fail {
    param([string]$Text)

    $Text = Protect-DiagnosticText $Text
    Write-Host "[FAIL] $Text" -ForegroundColor Red
    $script:Failures += $Text
}

function Write-Info {
    param([string]$Text)

    $Text = Protect-DiagnosticText $Text
    Write-Host "[INFO] $Text" -ForegroundColor Gray
}


# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------

function Test-CommandExists {
    param([string]$Name)

    return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

function Test-TcpPort {
    param(
        [string]$HostName,
        [int]$Port,
        [int]$TimeoutMs = 4000
    )

    $client = New-Object System.Net.Sockets.TcpClient

    try {
        $iar = $client.BeginConnect($HostName, $Port, $null, $null)

        if (-not $iar.AsyncWaitHandle.WaitOne($TimeoutMs, $false)) {
            $client.Close()
            return $false
        }

        $client.EndConnect($iar)
        $client.Close()

        return $true
    }
    catch {
        try {
            $client.Close()
        }
        catch {}

        return $false
    }
}

function Get-ProxyPortFromUrl {
    param([string]$ProxyValue)

    if ([string]::IsNullOrWhiteSpace($ProxyValue)) {
        return $null
    }

    try {
        $ProxyValue = $ProxyValue.Trim()
        if ($ProxyValue -match '^(http|https|ftp|socks)\s*=\s*(.+)$') {
            $proxyAssignment = $matches[1]
            $ProxyValue = $matches[2].Trim()
            if ($proxyAssignment -eq 'socks' -and $ProxyValue -notmatch '^[a-zA-Z][a-zA-Z0-9+.-]*://') {
                $ProxyValue = "socks4://$ProxyValue"
            }
        }
        if ($ProxyValue -notmatch "^[a-zA-Z][a-zA-Z0-9+.-]*://") {
            $ProxyValue = "http://$ProxyValue"
        }

        $uri = [Uri]$ProxyValue
        if (-not $uri.IsAbsoluteUri -or [string]::IsNullOrWhiteSpace($uri.Host)) {
            return $null
        }
        $port = $uri.Port
        if ($port -lt 1 -and $uri.Scheme -match '^socks(?:4a?|5h?)$') { $port = 1080 }
        if ($port -lt 1 -or $port -gt 65535) { return $null }
        $proxyHostName = $uri.Host.Trim('[', ']')
        $authority = if ($proxyHostName.Contains(':')) { "[$proxyHostName]" } else { $proxyHostName }

        return [PSCustomObject]@{
            Host = $proxyHostName
            Port = $port
            Uri  = $uri.AbsoluteUri
            ConnectUri = '{0}://{1}:{2}' -f $uri.Scheme, $authority, $port
            HasCredentials = -not [string]::IsNullOrWhiteSpace($uri.UserInfo)
        }
    }
    catch {
        return $null
    }
}

function Get-ProxyEntries {
    param([string]$Value)
    # A single WinINET assignment is valid too; do not split credentials in a URI.
    $parts = if ($Value -match '^\s*(?:http|https|ftp|socks)\s*=') {
        $Value -split ';'
    } else { @($Value) }
    foreach ($part in $parts) {
        $parsed = Get-ProxyPortFromUrl $part
        if ($parsed) { $parsed }
    }
}

function Format-EndpointForDisplay {
    param([string]$Value)
    if ([string]::IsNullOrWhiteSpace($Value)) { return '(not set)' }
    $entries = @(Get-ProxyEntries $Value)
    if (-not $entries.Count) { return '<configured value hidden; unrecognized endpoint>' }
    return (($entries | ForEach-Object { $_.ConnectUri }) -join '; ')
}

function Protect-DiagnosticText {
    param([string]$Text)
    # Retain endpoint identity, never URL userinfo, paths, query tokens or fragments.
    $Text = [regex]::Replace($Text, '[a-zA-Z][a-zA-Z0-9+.-]*://[^\s"''<>]+', {
        param($match)
        Format-EndpointForDisplay $match.Value
    })
    $Text = [regex]::Replace($Text, '[^\s/@]+@([^\s/]+)', '<redacted>@$1')
    return [regex]::Replace($Text,
        '(?i)\b(authorization|api[_-]?key|access[_-]?token|password|secret)\s*[:=]\s*[^\s,;]+',
        '$1=<redacted>')
}

function Test-RepeatedLoopbackFailure {
    param($Endpoint)
    if (-not $Endpoint) { return $false }
    $address = $null
    $loopback = $Endpoint.Host -ieq 'localhost'
    if ([Net.IPAddress]::TryParse($Endpoint.Host.Trim('[', ']'), [ref]$address)) {
        $loopback = [Net.IPAddress]::IsLoopback($address)
    }
    if (-not $loopback) { return $false }
    for ($attempt = 0; $attempt -lt 3; $attempt++) {
        if (Test-TcpPort $Endpoint.Host $Endpoint.Port 2000) { return $false }
        if ($attempt -lt 2) { Start-Sleep -Milliseconds 500 }
    }
    return $true
}

function Remove-UnreachableLocalConfiguration {
    param([string]$Path, [ValidateSet('env', 'config')][string]$Kind)
    # Every file operation must stop on failure; never write after a failed read
    # or backup. Keep original line endings and BOM/encoding when dropping lines.
    $fullPath = (Get-Item -LiteralPath $Path -ErrorAction Stop).FullName
    $bytes = [IO.File]::ReadAllBytes($fullPath)
    $stream = New-Object IO.MemoryStream(,$bytes)
    $reader = New-Object IO.StreamReader($stream, (New-Object Text.UTF8Encoding($false, $true)), $true)
    try {
        $original = $reader.ReadToEnd()
        $encoding = $reader.CurrentEncoding
    } finally { $reader.Dispose(); $stream.Dispose() }
    $retained = New-Object Text.StringBuilder
    $removed = 0
    foreach ($line in [regex]::Split($original, '(?<=\n)')) {
        $endpoint = $null
        if ($Kind -eq 'env' -and $line -match '^\s*(?:HTTP_PROXY|HTTPS_PROXY|ALL_PROXY)\s*=\s*(.+)$') {
            $endpoint = Get-ProxyPortFromUrl ($matches[1].Trim().Trim('"').Trim("'"))
        } elseif ($Kind -eq 'config' -and $line -match '^\s*openai_base_url\s*=\s*["''](.+?)["'']') {
            $endpoint = Get-ProxyPortFromUrl $matches[1]
        }
        if (Test-RepeatedLoopbackFailure $endpoint) { $removed++ }
        else { [void]$retained.Append($line) }
    }
    if (-not $removed) { return $false }
    New-Item -Path $BackupRoot -ItemType Directory -Force -ErrorAction Stop | Out-Null
    $backup = Join-Path $BackupRoot ((Split-Path $fullPath -Leaf) + '.' + [Guid]::NewGuid().ToString('N'))
    Copy-Item -LiteralPath $fullPath -Destination $backup -ErrorAction Stop
    $temporary = Join-Path (Split-Path $fullPath -Parent) ('.network-doctor-' + [Guid]::NewGuid().ToString('N') + '.tmp')
    try {
        if ([Convert]::ToBase64String([IO.File]::ReadAllBytes($fullPath)) -ne [Convert]::ToBase64String($bytes) -or
            [Convert]::ToBase64String([IO.File]::ReadAllBytes($backup)) -ne [Convert]::ToBase64String($bytes)) {
            throw 'Configuration changed during diagnosis; removal cancelled.'
        }
        [IO.File]::WriteAllText($temporary, $retained.ToString(), $encoding)
        [IO.File]::Replace($temporary, $fullPath, $null)
    } finally {
        if (Test-Path -LiteralPath $temporary -ErrorAction Stop) {
            Remove-Item -LiteralPath $temporary -ErrorAction Stop
        }
    }
    Write-Info "Backup: $backup"
    Write-Pass "Removed $removed repeatedly unreachable loopback entries from $fullPath"
    return $true
}

function Test-CurlEndpoint {
    param([string]$Target, $Proxy)
    $route = 'OS route; explicit proxy disabled (TUN/VPN may still apply)'
    $curlArguments = @('-q', '--head', '--connect-timeout', '10', '--max-time', '15',
        '--silent', '--show-error', '--output', 'NUL', '--write-out',
        'CODEX_NET http=%{http_code} connect=%{time_connect} tls=%{time_appconnect} total=%{time_total}')
    if ($Proxy) {
        if ($Proxy.HasCredentials) {
            Write-Warn "Authenticated proxy $(Format-EndpointForDisplay $Proxy.Uri): HTTP probe skipped to avoid exposing credentials in process arguments."
            return
        }
        $route = 'explicit proxy ' + (Format-EndpointForDisplay $Proxy.Uri)
        # A nonmatching bypass name overrides an inherited NO_PROXY without
        # passing an empty native argument (which Windows PowerShell can drop).
        $curlArguments += @('--proxy', $Proxy.ConnectUri, '--noproxy', 'codex-network-doctor.invalid')
    } else {
        $curlArguments += @('--noproxy', '*')
    }
    $curlArguments += $Target
    $curlResult = @(& curl.exe @curlArguments 2>&1)
    $exitCode = $LASTEXITCODE
    $joined = $curlResult -join ' '
    $httpStatus = 0
    if ($joined -match 'CODEX_NET http=(\d{3}) connect=([0-9.]+) tls=([0-9.]+) total=([0-9.]+)') {
        $httpStatus = [int]$matches[1]
        Write-Info "$Target [$route]: HTTP $httpStatus; TCP $($matches[2])s, TLS $($matches[3])s, total $($matches[4])s"
    }
    if ($exitCode -eq 0) {
        Write-Pass "TLS transport completed: $Target [$route]"
        if ($httpStatus -ge 400 -or $httpStatus -eq 0) {
            Write-Warn "HTTP $httpStatus is not application success; authentication, rate limits or access policy may still block requests."
        } else {
            Write-Info 'HTTP response received; authenticated application/session connectivity was not tested.'
        }
    } else {
        Write-Fail "HTTPS transport failed: $Target [$route] (curl exit=$exitCode)"
        # Do not print raw curl errors: they can echo URLs or proxy credentials.
        if ($joined -match 'CRYPT_E_REVOCATION_OFFLINE|revocation|schannel') {
            Write-Warn 'Curl reported a Windows TLS/certificate error; certificate validation remains enabled.'
        }
    }
}


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

Write-Title "Codex Network Doctor"

Write-Info "Time      : $(Get-Date)"
Write-Info "Windows   : $([Environment]::OSVersion.VersionString)"
Write-Info "PowerShell: $($PSVersionTable.PSVersion)"
Write-Info "User Home : $UserHome"
Write-Info "CODEX_HOME: $CodexHome"
Write-Info "Repair    : $Repair"
Write-Info "Remove unreachable local entries: $RemoveUnreachableLocalProxy"
Write-Info "Deep      : $Deep"
if ($RemoveUnreachableLocalProxy -and -not $Repair) {
    throw '-RemoveUnreachableLocalProxy must be combined with -Repair.'
}


# ------------------------------------------------------------
# 1. DNS
# ------------------------------------------------------------

Write-Section "1. DNS"

foreach ($target in $Targets) {
    try {
        $addresses = [System.Net.Dns]::GetHostAddresses($target)

        if ($addresses.Count -gt 0) {
            $ips = ($addresses | ForEach-Object { $_.IPAddressToString }) -join ", "
            Write-Pass "$target -> $ips"
        }
        else {
            Write-Fail "$target DNS returned no address"
        }
    }
    catch {
        Write-Fail "$target DNS failed: $($_.Exception.Message)"
    }
}


# ------------------------------------------------------------
# 2. TCP
# ------------------------------------------------------------

Write-Section "2. TCP 443 via OS routing (includes active TUN/VPN)"

foreach ($target in $Targets) {
    if (Test-TcpPort -HostName $target -Port 443) {
        Write-Pass "$target`:443 reachable"
    }
    else {
        Write-Fail "$target`:443 unreachable"
    }
}


# ------------------------------------------------------------
# 3. HTTPS / TLS using curl
# ------------------------------------------------------------

Write-Section "3. HTTPS / TLS"

if (Test-CommandExists "curl.exe") {
    Write-Info 'No-proxy probes disable explicit HTTP/SOCKS proxying only; they do not bypass Clash TUN, VPNs or OS routes.'
    $httpProxies = @()
    foreach ($name in @('HTTPS_PROXY', 'HTTP_PROXY', 'ALL_PROXY')) {
        $httpProxies += @(Get-ProxyEntries ([Environment]::GetEnvironmentVariable($name, 'Process')))
    }
    try {
        $httpsSettings = Get-ItemProperty -LiteralPath 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings' -ErrorAction Stop
        if ($httpsSettings.ProxyEnable -eq 1) {
            $httpProxies += @(Get-ProxyEntries $httpsSettings.ProxyServer)
        }
        if ($httpsSettings.AutoConfigURL) { Write-Info 'PAC is configured; this curl comparison does not evaluate PAC rules.' }
    } catch { Write-Warn 'Unable to collect WinINET proxy for HTTPS comparison.' }
    $httpProxies = @($httpProxies | Sort-Object Uri -Unique)
    foreach ($target in @('https://chatgpt.com', 'https://api.openai.com')) {
        Test-CurlEndpoint -Target $target
        foreach ($proxy in $httpProxies) { Test-CurlEndpoint -Target $target -Proxy $proxy }
    }
}
else {
    Write-Warn "curl.exe not found"
}


# ------------------------------------------------------------
# 4. Environment proxy
# ------------------------------------------------------------

Write-Section "4. Process/User/System Proxy Environment"

$ProxyVariables = @(
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY"
)

$FoundProxyValues = @()

foreach ($name in $ProxyVariables) {

    $processValue = [Environment]::GetEnvironmentVariable($name, "Process")
    $userValue = [Environment]::GetEnvironmentVariable($name, "User")
    $machineValue = [Environment]::GetEnvironmentVariable($name, "Machine")

    if ($processValue) {
        $display = if ($name -eq 'NO_PROXY') { '<bypass list configured; value hidden>' } else { Format-EndpointForDisplay $processValue }
        Write-Info "$name [Process] = $display"

        if ($name -notmatch "NO_PROXY") {
            $FoundProxyValues += $processValue
        }
    }

    if ($userValue) {
        $display = if ($name -eq 'NO_PROXY') { '<bypass list configured; value hidden>' } else { Format-EndpointForDisplay $userValue }
        Write-Info "$name [User] = $display"

        if ($name -notmatch "NO_PROXY") {
            $FoundProxyValues += $userValue
        }
    }

    if ($machineValue) {
        $display = if ($name -eq 'NO_PROXY') { '<bypass list configured; value hidden>' } else { Format-EndpointForDisplay $machineValue }
        Write-Info "$name [Machine] = $display"

        if ($name -notmatch "NO_PROXY") {
            $FoundProxyValues += $machineValue
        }
    }
}

if ($FoundProxyValues.Count -eq 0) {
    Write-Pass "No HTTP_PROXY/HTTPS_PROXY/ALL_PROXY environment variables detected"
}


# ------------------------------------------------------------
# 5. WinINET Proxy
# ------------------------------------------------------------

Write-Section "5. Windows WinINET Proxy"

$InternetSettings =
    "HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings"

try {

    $settings = Get-ItemProperty $InternetSettings

    Write-Info "ProxyEnable = $($settings.ProxyEnable)"
    Write-Info "ProxyServer = $(Format-EndpointForDisplay $settings.ProxyServer)"
    Write-Info "AutoConfigURL = $(Format-EndpointForDisplay $settings.AutoConfigURL)"

    if ($settings.ProxyEnable -eq 1) {
        Write-Info "Windows WinINET Proxy enabled: $(Format-EndpointForDisplay $settings.ProxyServer)"

        if ($settings.ProxyServer) {
            $FoundProxyValues += $settings.ProxyServer
        }
    }
    else {
        Write-Pass "WinINET explicit proxy disabled"
    }

}
catch {
    Write-Warn "Unable to read WinINET proxy: $($_.Exception.Message)"
}


# ------------------------------------------------------------
# 6. WinHTTP Proxy
# ------------------------------------------------------------

Write-Section "6. Windows WinHTTP Proxy"

try {
    $winHttp = & netsh winhttp show proxy 2>&1
    if ($LASTEXITCODE -ne 0) { throw 'netsh returned a failure exit code.' }
    $winHttp | ForEach-Object {
        Write-Info (Protect-DiagnosticText ([string]$_))
    }
}
catch {
    Write-Warn "netsh winhttp show proxy failed"
}


# ------------------------------------------------------------
# 7. Detect local proxy ports
# ------------------------------------------------------------

Write-Section "7. Local Proxy Port"

$KnownProxyPairs = @()

foreach ($value in ($FoundProxyValues | Select-Object -Unique)) {
    $KnownProxyPairs += @(Get-ProxyEntries $value)
}


foreach ($port in $CommonProxyPorts) {
    $KnownProxyPairs += [PSCustomObject]@{
        Host = $ProxyHost
        Port = $port
        Uri  = "$ProxyHost`:$port"
    }
}


$KnownProxyPairs =
    $KnownProxyPairs |
    Sort-Object Host, Port -Unique


foreach ($proxy in $KnownProxyPairs) {

    if (Test-TcpPort -HostName $proxy.Host -Port $proxy.Port -TimeoutMs 1000) {
        Write-Info "TCP listener detected: $($proxy.Host):$($proxy.Port) (proxy protocol not established by this check)"
    }
    else {
        Write-Info "No listener: $($proxy.Host):$($proxy.Port)"
    }
}


# ------------------------------------------------------------
# 8. ~/.codex/.env
# ------------------------------------------------------------

Write-Section "8. Codex .env"

$CodexEnvHasBadProxy = $false

if (Test-Path -LiteralPath $CodexEnvFile) {

    Write-Info "Found: $CodexEnvFile"

    $envLines = try { Get-Content -LiteralPath $CodexEnvFile -ErrorAction Stop }
                catch { Write-Warn 'Unable to read the Codex .env file; leaving it unchanged.'; @() }

    foreach ($line in $envLines) {

        if ($line -match "^\s*(HTTP_PROXY|HTTPS_PROXY|ALL_PROXY|http_proxy|https_proxy|all_proxy)\s*=\s*(.+)$") {

            $proxyName = $matches[1]
            $proxyValue = $matches[2].Trim().Trim('"').Trim("'")

            Write-Info "$proxyName = $(Format-EndpointForDisplay $proxyValue)"

            $proxy = Get-ProxyPortFromUrl $proxyValue

            if ($proxy) {

                if (Test-TcpPort $proxy.Host $proxy.Port 1200) {
                    Write-Pass "Codex .env proxy listener active: $($proxy.Host):$($proxy.Port)"
                }
                else {
                    Write-Warn "Codex .env proxy was unreachable during this check: $(Format-EndpointForDisplay $proxyValue)"
                    $CodexEnvHasBadProxy = $true
                }
            }
        }
    }

}
else {
    Write-Pass "No $CodexEnvFile"
}


# ------------------------------------------------------------
# 9. config.toml
# ------------------------------------------------------------

Write-Section "9. Codex config.toml"

$BadBaseUrl = $false

if (Test-Path -LiteralPath $CodexConfigFile) {

    Write-Info "Found: $CodexConfigFile"

    $configLines = try { Get-Content -LiteralPath $CodexConfigFile -ErrorAction Stop }
                   catch { Write-Warn 'Unable to read config.toml; leaving it unchanged.'; @() }

    foreach ($line in $configLines) {

        if ($line -match "^\s*openai_base_url\s*=\s*[`"'](.+?)[`"']") {

            $baseUrl = $matches[1]

            Write-Info "openai_base_url = $(Format-EndpointForDisplay $baseUrl)"

            try {

                $uri = [Uri]$baseUrl

                if (
                    $uri.Host -eq "127.0.0.1" -or
                    $uri.Host -eq "localhost"
                ) {

                    if (-not (Test-TcpPort $uri.Host $uri.Port 1200)) {
                        Write-Warn "openai_base_url local endpoint was unreachable during this check: $(Format-EndpointForDisplay $baseUrl)"
                        $BadBaseUrl = $true
                    }
                    else {
                        Write-Pass "openai_base_url local endpoint is listening"
                    }

                }
            }
            catch {
                Write-Warn 'Invalid openai_base_url; value hidden.'
                $BadBaseUrl = $true
            }
        }
    }

}
else {
    Write-Pass "No custom config.toml found"
}


# ------------------------------------------------------------
# 10. Codex / Node processes
# ------------------------------------------------------------

Write-Section "10. Codex / Node Processes"

$interestingProcesses = @()

try {
    $interestingProcesses =
        Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -match "codex|node|chatgpt" -or
            $_.CommandLine -match "codex"
        }

    if ($interestingProcesses.Count -eq 0) {
        Write-Info "No Codex related process currently detected"
    }
    else {

        foreach ($proc in $interestingProcesses) {

            Write-Info (
                "PID={0} Name={1}" -f
                $proc.ProcessId,
                $proc.Name
            )

            if ($Deep -and $proc.CommandLine) {
                Write-Info 'Command line present; hidden because arguments can contain credentials.'
            }
        }

        $codexCount =
            ($interestingProcesses |
            Where-Object {
                $_.Name -match "codex" -or
                $_.CommandLine -match "codex"
            }).Count

        Write-Info "Codex-related process count: $codexCount (multiple helper processes are not a network failure)."
    }

}
catch {
    Write-Warn "Unable to inspect process command lines"
}


# ------------------------------------------------------------
# 11. Local Codex storage
# ------------------------------------------------------------

Write-Section "11. Codex Local Storage"

if (Test-Path $CodexHome) {

    try {

        $largeFiles =
            Get-ChildItem $CodexHome -File -Recurse -ErrorAction SilentlyContinue |
            Where-Object {
                $_.Length -gt 100MB
            } |
            Sort-Object Length -Descending |
            Select-Object -First 15

        if ($largeFiles) {

            Write-Warn "Large files detected under CODEX_HOME"

            foreach ($file in $largeFiles) {
                Write-Host (
                    "       {0,8:N1} MB  {1}" -f
                    ($file.Length / 1MB),
                    $file.FullName
                ) -ForegroundColor DarkYellow
            }

        }
        else {
            Write-Pass "No >100MB files detected under CODEX_HOME"
        }

    }
    catch {
        Write-Warn "Unable to scan CODEX_HOME"
    }

}
else {
    Write-Info "CODEX_HOME does not exist yet"
}


# ------------------------------------------------------------
# 12. Deep checks
# ------------------------------------------------------------

if ($Deep) {

    Write-Section "12. Deep Network Checks"

    Write-Info "Route table:"
    try {
        Get-NetRoute |
            Where-Object {
                $_.DestinationPrefix -eq "0.0.0.0/0"
            } |
            Format-Table -AutoSize
    }
    catch {}

    Write-Info "DNS servers:"
    try {
        Get-DnsClientServerAddress |
            Where-Object {
                $_.ServerAddresses.Count -gt 0
            } |
            Format-Table InterfaceAlias, AddressFamily, ServerAddresses -AutoSize
    }
    catch {}

    Write-Info "Active TCP connections to 443:"
    try {
        Get-NetTCPConnection -State Established -ErrorAction SilentlyContinue |
            Where-Object {
                $_.RemotePort -eq 443
            } |
            Select-Object -First 30 |
            Format-Table LocalAddress, LocalPort, RemoteAddress, RemotePort, OwningProcess
    }
    catch {}
}


# ------------------------------------------------------------
# REPAIR
# ------------------------------------------------------------

if ($Repair) {

    Write-Title "Safe Repair"

    $anythingChanged = $false
    $configurationChanged = $false
    if ($RemoveUnreachableLocalProxy) {
        foreach ($entry in @(@{ Path = $CodexEnvFile; Kind = 'env' },
                             @{ Path = $CodexConfigFile; Kind = 'config' })) {
            try {
                if (Test-Path -LiteralPath $entry.Path -PathType Leaf -ErrorAction Stop) {
                    if (Remove-UnreachableLocalConfiguration -Path $entry.Path -Kind $entry.Kind) {
                        $configurationChanged = $true
                        $anythingChanged = $true
                    }
                }
            } catch {
                Write-Fail "Local configuration cleanup stopped: $($_.Exception.Message)"
            }
        }
    } else {
        Write-Info 'Persistent proxy and endpoint settings are preserved. -Repair only flushes DNS.'
    }


    # --------------------------------------------------------
    # Flush DNS
    # --------------------------------------------------------

    try {
        ipconfig /flushdns | Out-Null
        if ($LASTEXITCODE -ne 0) { throw 'ipconfig returned a failure exit code.' }
        Write-Pass "DNS cache flushed"
        $anythingChanged = $true
    }
    catch {
        Write-Warn "Failed to flush DNS"
    }


    # --------------------------------------------------------
    # We deliberately do NOT reset winsock automatically.
    # --------------------------------------------------------

    Write-Info "Winsock reset NOT performed automatically."
    Write-Info "Proxy registry NOT modified automatically."
    Write-Info "Certificate settings NOT modified automatically."

    if ($anythingChanged) {

        Write-Host ""
        Write-Host "Changes completed." -ForegroundColor Green

        if (Test-Path -LiteralPath $BackupRoot) {
            Write-Host "Backup directory:" -ForegroundColor Cyan
            Write-Host "  $BackupRoot"
        }

        Write-Host ""
        if ($configurationChanged) {
            Write-Warn 'Local configuration changed; restart the affected application when ready.'
        }
    }
    else {
        Write-Info "No safe automatic repair was necessary."
    }
}


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

Write-Title "Diagnosis Summary"

Write-Host "PASS : $($script:Passes.Count)" -ForegroundColor Green
Write-Host "WARN : $($script:Warnings.Count)" -ForegroundColor Yellow
Write-Host "FAIL : $($script:Failures.Count)" -ForegroundColor Red

Write-Host ""

if ($script:Failures.Count -eq 0) {

    Write-Host "No obvious basic network failure was detected." -ForegroundColor Green

    Write-Host ""
    Write-Host "If Codex still keeps reconnecting, the priority suspects are:" `
        -ForegroundColor Yellow

    Write-Host "  1. WebSocket blocked/intercepted by proxy"
    Write-Host "  2. Proxy implementation compatibility issue"
    Write-Host "  3. Codex local session/state issue"
    Write-Host "  4. Security software doing HTTPS/TLS inspection"

}
else {

    Write-Host "Detected failures:" -ForegroundColor Red

    foreach ($failure in $script:Failures) {
        Write-Host "  - $failure"
    }
}


Write-Host ""

if ($CodexEnvHasBadProxy) {
    Write-Warn 'An .env proxy was unreachable during the initial probe; a stopped proxy or transient failure is not proof of a bad configuration.'
}

if ($BadBaseUrl) {
    Write-Warn 'An openai_base_url issue was observed during the initial probe; review the local service before changing its endpoint.'
}

Write-Host ""
Write-Host "Optional DNS-cache repair (does not delete configuration):" -ForegroundColor Cyan

if (-not $Repair) {
    Write-Host (
        "powershell -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Repair"
    )
}
else {
    Write-Host "Run diagnosis again to measure the current state."
}

Write-Host ""
