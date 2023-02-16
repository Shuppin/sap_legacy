# Changelog
Format: `[version] - DD-MM-YYYY`
#

## [0.0.1-pre.43] - 16-02-2023

Pretty big update here

### Added

- `while` statements
- `inc` and `dec` statements
- `loops.sap` to demonstrate while loops
- `increments.sap` to demonstrate `inc` and `dec`
- Quickly made skeleton code allowing the error printer to accept a list of tokens as an input. It's likely that this code is riddled with error so i'll update in the future since it's not needed for now
- `get_nth_token()` method to `Node`, this function allows the code to search for a token in a Node and it's children. Useful for error printing

### Changed

- Fixed a major bug in `variable_assignment()` where a `Token` was being passed instead of a `VarNode`, causing the program to fail later on in `visit_AssignOp()` (Interpreter)
- Fixed a major bug where program would crash if an expression passed into a selection statement couldn't be evaluated as a boolean
- Some general code cleanup as per usual
- Newlines now act as seperators, in other words, lines do not need to be followed by a semicolon anymore (Though you still can if you want to)
- Replaced `def` keyword with `fn` because it looks to much like python lol
- Minor changes to to lexer to corectly identify the position of operators which are longer than 1 character
- Dates were all wrong in the changelog, fixed now

### Removed

- Old code which perfomed semicolon checks
- Along with this, the `strict_semicolon` option in `config.py`

## [0.0.1-pre.42] - 12-02-2023

### Added

- Selection is now fully implemented
- A few comments in the interpreter

### Changed

- Slight logic rework on assign operations

## [0.0.1-pre.41] - 10-02-2023

### Changed

- Simplified binary operation code

### Added

- Lexing & Parsing for if, elseif and else statements (Testing still needs to be done)

## [0.0.1-pre.40] - 08-02-2023

### Changed

- Completely rewrote arithmetic, logic and comparsion system to use less syntax. `logic.py`, `arithmetic.py` and `comparison.py` have been reduced to a single file `operand.py`
- Cleaned up code and improved documentation in certain areas
- Comparsion operator `!=` (not equal to) changed to `~=`

### Removed

- `overloading.py` module no longer used, moved to `SAP/other`

## [0.0.1-pre.39] - 30-01-2023

### Added

- `comparison.py` module, still needs finishing

### Changed

- `syntax_showcase.sap` & `procedure_calls.sap` now execute fully

## [0.0.1-pre.38] - 30-01-2023

### Added

- Comparison operators parsing (need to update interpreter)

## [0.0.1-pre.37] - 26-01-2023

### Changes

- Replaced `:=` symbol with `=`
- Updated `__version__` (i forgot it existed)

### Added

- Format information to changelog file

## [0.0.1-pre.36] - 25-01-2023

### Added

- Boolean logic operations

### Changes

- Minor bug fixes & code clean up


## [0.0.1-pre.35] - 25-01-2023

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