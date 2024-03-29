# Changelog
Format: `[version] - DD-MM-YYYY`
#

## [1.0.0] - 12-09-2023

What is that... a full release!?!

Well... kinda.

It seems this project has plateaued. I have largely achieved my goals for this language, and don't intend to make many updates for in the future. 

See you all when SAP v2.0 comes out!

### Added

- Read me files providing information to whoever may stumble across this ancient artifact.

### Changes

- Made GitHub repo public.

## [0.0.1-pre.47] - 27-03-2023

### Changes

- Fixed major bug in error printer cause program to crash because it was incorrectly selecting token index
- Various improvements to error messages
- Reformatted `changelog.md` replacing 'changed' with 'changes'

## [0.0.1-pre.46] - 21-03-2023

### Added

- Modulus `%` operator, order of operations lies before addition, but after multiplication

### Changes

- Revised example code, much better structure and organisation now
- Small changes to logging, mostly wording and log levels
- Tried to fix error printer, solved one problem and created 5 more :)
- Fixed a problem where the incorrect variable type was being passed into the symbol table inside a function
- Very quick fix to prevent the program from crashing upon zero division by just returning 0 (I know this is awful it's only temporary, i hope)


## [0.0.1-pre.45] - 20-03-2023

Nearly a month of inactivity cause my laptop broke lol

### Added

- Built-in functions which link to actual python functions and can be executed from within the code!
- The only two functions right now are `print` and `println`
- Colours! Various messages printed to stdout now have colour (Windows only)
- `VarSymbol` now also keeps track of the variable type, not used right now but will (potentially) be used in future for semantic analysis

### Changes

- The built-in symbol table is now created explicity instead of inside of the semantic analyser, this allows more flexibility and lays the groundwork for a potential package/module system.
- Restructed error printing code to reduce number of lines
- Renamed some of the attributes of `BuiltinProcedureSymbol` to make more sense
- More spelling fixes

### Removed

- `BuiltinTypeSymbol` class, this is because types are now treated as part of the base syntax, making this code redundant.

## [0.0.1-pre.44] - 21-02-2023

### Added

- `string` datatype
- `strings.sap` to demonstrate syntax
- Some basic skeleton code to demonstrate a potential implementation of builtin methods
- Along with this, a `builtin_methods.py` which will eventually be used to store all built in methods

### Changes

- Everybody's favourite! Code cleanup!

## [0.0.1-pre.43] - 16-02-2023

Pretty big update here

### Added

- `while` statements
- `inc` and `dec` statements
- `loops.sap` to demonstrate while loops
- `increments.sap` to demonstrate `inc` and `dec`
- Quickly made skeleton code allowing the error printer to accept a list of tokens as an input. It's likely that this code is riddled with error so i'll update in the future since it's not needed for now
- `get_nth_token()` method to `Node`, this function allows the code to search for a token in a Node and it's children. Useful for error printing

### Changes

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

### Changes

- Slight logic rework on assign operations

## [0.0.1-pre.41] - 10-02-2023

### Changes

- Simplified binary operation code

### Added

- Lexing & Parsing for if, elseif and else statements (Testing still needs to be done)

## [0.0.1-pre.40] - 08-02-2023

### Changes

- Completely rewrote arithmetic, logic and comparsion system to use less syntax. `logic.py`, `arithmetic.py` and `comparison.py` have been reduced to a single file `operand.py`
- Cleaned up code and improved documentation in certain areas
- Comparsion operator `!=` (not equal to) changed to `~=`

### Removed

- `overloading.py` module no longer used, moved to `SAP/other`

## [0.0.1-pre.39] - 30-01-2023

### Added

- `comparison.py` module, still needs finishing

### Changes

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