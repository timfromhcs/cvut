param (
    [string]$Token = $env:GITHUB_TOKEN
)

if (-not $Token) {
    Write-Error "Please set GITHUB_TOKEN environment variable or pass -Token."
    exit 1
}

$headers = @{
    "Authorization" = "Bearer $Token"
    "Accept" = "application/vnd.github+json"
    "User-Agent" = "CVUT-CI-Watcher"
}

Write-Host "[WATCHER] Querying latest workflow runs..."
$runs = (Invoke-RestMethod -Uri "https://api.github.com/repos/timfromhcs/cvut/actions/runs?per_page=5" -Headers $headers).workflow_runs

if (-not $runs -or $runs.Count -eq 0) {
    Write-Host "[WATCHER] No runs found yet. Waiting 10s..."
    Start-Sleep -Seconds 10
    $runs = (Invoke-RestMethod -Uri "https://api.github.com/repos/timfromhcs/cvut/actions/runs?per_page=5" -Headers $headers).workflow_runs
}

$targetRun = $runs[0]
$runId = $targetRun.id
Write-Host "[WATCHER] Observing Run ID: $runId ($($targetRun.html_url)) [Event: $($targetRun.event)]"

while ($true) {
    $current = Invoke-RestMethod -Uri "https://api.github.com/repos/timfromhcs/cvut/actions/runs/$runId" -Headers $headers
    $jobsData = Invoke-RestMethod -Uri "https://api.github.com/repos/timfromhcs/cvut/actions/runs/$runId/jobs" -Headers $headers
    $jobs = $jobsData.jobs

    $statusSummary = @()
    foreach ($j in $jobs) {
        $statusSummary += "$($j.name): $($j.status) ($($j.conclusion))"
    }
    Write-Host "[WATCHER] Run Status: $($current.status) | Conclusion: $($current.conclusion)"
    Write-Host "          Jobs: $($statusSummary -join ' | ')"

    if ($current.status -eq "completed") {
        if ($current.conclusion -eq "success") {
            Write-Host "[WATCHER] SUCCESS: All matrix jobs passed successfully!" -ForegroundColor Green
            exit 0
        } else {
            Write-Host "[WATCHER] FAILURE: Run finished with conclusion: $($current.conclusion)" -ForegroundColor Red
            foreach ($j in $jobs) {
                if ($j.conclusion -eq "failure") {
                    Write-Host "[FAILED JOB] $($j.name) (ID: $($j.id))"
                    foreach ($s in $j.steps) {
                        if ($s.conclusion -eq "failure") {
                            Write-Host "  -> Failed Step: $($s.name)"
                        }
                    }
                }
            }
            exit 1
        }
    }

    Start-Sleep -Seconds 20
}
