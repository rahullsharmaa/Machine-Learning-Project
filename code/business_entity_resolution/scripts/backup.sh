#!/usr/bin/env bash
# Snapshot the pipeline source + small artifacts into persistent Files storage.
set -e
cd /dev-server
tar --exclude='__pycache__' -czf /mnt/documents/er_pipeline_backup.tar.gz er_pipeline
ls -la /mnt/documents/er_pipeline_backup.tar.gz
