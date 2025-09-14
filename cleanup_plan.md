# Step-by-step cleanup execution log

## Files to remove (confirmed empty/duplicate):
- CONTRIBUTING.md (empty, proper version exists in docs/)
- BCI_Validation_Framework_Summary.md (empty)
- EMG_UPDATE.md (empty)
- STATUS_REPORT.md (empty)
- README_NEW.md (obsolete)
- DOCKER_BUILD_FIX.md (duplicate, exists in docs/guides/)
- DOCKER_QUICK_START.md (duplicate, exists in docs/guides/)

## Files to move to scripts/tools/:
- reorganization_complete.sh (has content, status report)
- completion_summary.sh (utility script)
- execute_cleanup.sh (utility script)
- execute_final_cleanup.sh (utility script)
- execute_reorganization.sh (utility script)
- final_cleanup.sh (utility script)
- reorganize_root.sh (utility script)
- run_cleanup_now.sh (utility script)
- test_reorganization.sh (test script)
- verify_reorganization.sh (verification script)
- comprehensive_fix.sh (utility script)

## Files to move to scripts/setup/:
- setup.sh (setup script)

## Files to move to scripts/tools/ (test/setup):
- setup_and_test.sh (test script)
- setup_and_test.bat (Windows test script)

## Docker files to move to docker/ (as backups):
- Dockerfile → docker/Dockerfile.root-backup
- Dockerfile.frontend → docker/ (if not already there)
- Dockerfile.minimal → docker/ (if not already there)
- docker-compose.yml → docker/compose/docker-compose.root-backup.yml
- docker-compose.dev.yml → docker/compose/ (if not already there)
