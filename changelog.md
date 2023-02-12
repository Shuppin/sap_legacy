# Changelog
Format: `[version] - DD-MM-YYYY`
#

## [0.0.1-pre.41] - 22-01-2023

### Added

- Selection is now fully implemented
- A few comments in the interpreter

### Changed

- Slight logic rework on assign operations


## [0.0.1-pre.41] - 22-01-2023

### Changed

- Simplified binary operation code

### Added

- Lexing & Parsing for if, elseif and else statements (Testing still needs to be done)

## [0.0.1-pre.40] - 22-01-2023

### Changed

- Completely rewrote arithmetic, logic and comparsion system to use less syntax. `logic.py`, `arithmetic.py` and `comparison.py` have been reduced to a single file `operand.py`
- Cleaned up code and improved documentation in certain areas
- Comparsion operator `!=` (not equal to) changed to `~=`

### Removed

- `overloading.py` module no longer used, moved to `SAP/other`

## [0.0.1-pre.39] - 22-01-2023

### Added

- `comparison.py` module, still needs finishing

### Changed

- `syntax_showcase.sap` & `procedure_calls.sap` now execute fully

## [0.0.1-pre.38] - 22-01-2023

### Added

- Comparison operators parsing (need to update interpreter)

## [0.0.1-pre.37] - 22-01-2023

### Changes

- Replaced `:=` symbol with `=`
- Updated `__version__` (i forgot it existed)

### Added

- Format information to changelog file

## [0.0.1-pre.36] - 22-01-2023

### Added

- Boolean logic operations

### Changes

- Minor bug fixes & code clean up


## [0.0.1-pre.35] - 22-01-2023

### Added

- Fully implemented booleans
- Minor bugfixes
- `.log` files are now ignored

## [0.0.1-pre.34] - 22-01-2023

### Changes

- Fixed issue with modules logging to the incorrect file

## [0.0.1-pre.33] - 22-01-2023

### Changes

- Fixed typo in `.gitignore` (which is why it wasn't working)

## [0.0.1-pre.32] - 22-01-2023

### Added

- Checks to verify the destination of the `runtime.log` file

## [0.0.1-pre.31] - 22-01-2023

### Removed

- `e.py` test file

## [0.0.1-pre.30] - 22-01-2023

### Changes

- Still getting used to this versioning thing lol

## [0.0.1-pre.29] - 22-01-2023

### Removed

- `__pycache__` folders

## [0.0.1-pre.28] - 22-01-2023

### Added

- Changelog!
- `.gitignore` file
- Interpreter now supports variables from the outer scope

### Changes

- Moved other files into `modules` folder
- Moved `icons` folder into `other` folder