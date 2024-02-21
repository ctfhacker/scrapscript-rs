# Scrapscript-rs

Implementation of [scrapscript](https://scrapscript.org) in Rust.

Currently:

* Tokenizing an input program
* Parsing the tokens into a syntax tree
* (WIP) Evaluating the syntax tree

## Testing

```
cargo test
```

## Syntax Tree

```
a + b
. a = 1
. b = 2
```

[svg](./docs/dump.svg)
