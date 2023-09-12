# SAP v1.0

This is by no means a comprehensive document covering all aspects of this repository, it is simply a quick introduction.

I may decide to add to this documentation in the future, but there isn't much point given since the original goals of this project have largely been met and I have no future plans for it. (Other than SAP v2.0 - Coming soon!)

## Introduction

Welcome to SAP v1.0!

SAP is an interpreted programming language with a syntax that sits somewhere between C and Python. It is a very minimal language, consisting only of the most basic features.

Some major features are still lacking and/or do not exist at all. For example the standard library consists of 2 functions: `print` and `input`. Another major feature that's missing is a return statement, functions do not have the ability to return values. 

SAP is written in Python, as a consequence it quite slow and very unstable.

## Running

Running is extremely simple, you simply run `sap.py` and give the file you wish to execute as an argument. If no argument was provided, it will run the default file as defined in `config.toml`.

Code examples can be found in the `/examples` folder.

## Features

All of the following syntax features are demonstrated in the examples provided.

Syntax features:
- Comments (Multi-line & single line)
- Basic types (String, Int, Float, Bool, NoneType)
- Variable declaration
- A few different variable assignment methods
- Arithmetic operators (Fundamental arithmetic, negation) 
- Boolean operators (not, and, or)
- Comparison operators
- Increment/decrement operators
- Standard order of operations (And brackets)
- Compound statements
- Built-in functions
- User-defined functions
- Nested functions
- While loops
- Control flow (if/elseif/else)

Technical features:
- Pretty verbose logging
- Config files
- Token streams
- Symbol tables
- Abstract syntax trees & tree nodes
- Tree traversal algorithms
- Type inference
- Stack frames & activation records


## Layout

The bulk of the program is contained within `sap.py` and various helper modules and class declarations can be found under the `/modules` folder.

The `sap.py` file is mostly laid out in the order of execution, with the entry point being at the very bottom.

## Design

It's design is that of a standard interpreter. The program consists of 4 main components:
- Lexical Analyser
- Parser
- Semantic Analyser
- Runtime

The Driver class is responsible for creating and running all of these components.

There are a few other components like the config loader and logging system.

Deeper within the code, there are some design choices which are less typical for a standard interpreter. These section are the result of trying to get certain features to work and, to be honest, are quite strange.

## Execution

You can get a good idea of what the program is doing behind the scenes but setting the logging level to max (`logging.level` should already be set to "ALL" in `config.toml`) and running one of the longer example files. 

In the log file you will be able to see almost every aspect of the program, including:
- The initialisation of each component and it's input data
- The creation and insertion of the standard library into the namespace
- The parsing process (AST creation) and token stream generation
- The AST itself
- Traversal of the AST
- The semantic analysis process involving:
  * Type checking
  * Symbol table stack creation
  * Pretty little symbol table debug printouts
  * Scope lookups
- The execution process involving:
  * Activation record printouts with information about the variable values and types in the current scope.
  * Call stack operations.

Note that these logs files can get very very long when logging at such a verbose level. This also has a significant effect on execution speed, as the log file is written to in realtime.
