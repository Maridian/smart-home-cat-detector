# Changelog

All notable changes to this project will be documented in this file.

## [2.0.0] - Refactored Release

### Added
- Central CLI interface via `main.py`
- Unified command system for all operations
- Improved utility modules (`src/utils/config.py`, `src/utils/device.py`)
- Comprehensive README with clear usage instructions
- Better error handling and user feedback
- Device information display before operations

### Changed
- All scripts now use centralized configuration
- Consistent English language across codebase
- Simplified project structure
- Improved console output formatting
- Better path management using utility functions

### Removed
- Individual PowerShell scripts (replaced by central CLI)
- Redundant code duplication across modules
- Unused benchmark utility in src/utils

### Improved
- Code organization and modularity
- Error messages and user guidance
- Documentation and inline comments
- Configuration management via .env
- Device selection (CPU/GPU) handling

## [1.0.0] - Initial Release

### Features
- RTSP stream data collection
- Automatic labeling with YOLOv8m
- Custom YOLOv8n training
- Model validation
- Performance benchmarking
- Live detection on RTSP streams
