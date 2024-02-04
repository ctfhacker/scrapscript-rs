use core::ops::{Index, IndexMut};
use std::collections::HashSet;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TokenId {
    Name,
    Int,
    Float,
    String,
    Comment,
    LeftParen,
    RightParen,
    LeftBrace,
    RightBrace,
    LeftBracket,
    RightBracket,
    Base85,
    Base64,
    Base32,
    Base16,
    Comma,
    Symbol,

    /// Operator .
    OpWhere,

    /// Operator =
    OpAssign,

    /// Operator |
    OpMatchCase,

    /// Operator ->
    OpFunction,

    /// Operator +
    OpAdd,

    /// Operator -
    OpSub,

    /// Operator ==
    OpEqual,

    /// Operator %
    OpMod,

    /// Operator //
    OpFloorDiv,

    /// Operator /=
    OpNotEqual,

    /// Operator <
    OpLess,

    /// Operator >
    OpGreater,

    /// Operator <=
    OpLessEqual,

    /// Operator >=
    OpGreaterEqual,

    /// Operator &&
    OpBoolAnd,

    /// Operator ||
    OpBoolOr,

    /// Operator ++
    OpStrConcat,

    /// Operator >+ (List concat)
    OpListCons,

    /// Operator >+ (List append)
    OpListAppend,

    /// Operator !
    OpRightEval,

    /// Operator :
    OpHasType,

    /// Operator |>
    OpPipe,

    /// Operator <|
    OpReversePipe,

    /// Operator ...
    OpSpread,

    /// Operator ?
    OpAssert,

    /// Operator *
    OpMul,

    /// Operator /
    OpDiv,

    /// Operator ^
    OpExp,

    /// Operator >>
    OpCompose,

    /// Operator <<
    OpComposeReverse,

    /// Operator @
    OpAccess,

    OpNegate,

    /// EOF
    EndOfFile,
}

impl TokenId {
    pub fn get_op_prescedence(&self) -> Option<u32> {
        use TokenId::*;

        match self {
            OpAccess => Some(1001),
            OpComposeReverse | OpCompose => Some(140),
            OpExp => Some(130),
            OpMul | OpDiv | OpFloorDiv | OpMod => Some(120),
            OpSub | OpAdd => Some(110),
            OpListCons | OpListAppend | OpStrConcat => Some(100),
            OpEqual | OpNotEqual | OpLess | OpGreater | OpLessEqual | OpGreaterEqual => Some(90),
            OpBoolAnd => Some(80),
            OpBoolOr => Some(70),
            OpPipe | OpReversePipe => Some(60),
            OpFunction => Some(50),
            OpMatchCase => Some(42),
            OpHasType => Some(41),
            OpAssign => Some(40),
            OpRightEval | OpWhere | OpAssert => Some(30),
            Comma => Some(10),
            OpSpread => Some(0),
            x => {
                println!("Unknown operator prescedence: {x:?}");
                return None;
            }
        }
    }
}

/// Abstract syntax tree nodes
#[derive(Debug, Clone, PartialEq)]
pub enum Node {
    /// An integer value (leaf)
    Int { data: i64 },

    /// A float value (leaf)
    Float { data: f64 },

    /// A variable
    Var { data: String },

    /// A vec of bytes
    Bytes { data: Vec<u8> },

    /// A exponentiation operation
    Exp { left: NodeId, right: NodeId },

    /// A subtraction operation between two nodes
    Sub { left: NodeId, right: NodeId },

    /// An addition operation between two nodes
    Add { left: NodeId, right: NodeId },

    /// A multiplication operation between two nodes
    Mul { left: NodeId, right: NodeId },

    /// A greater than operation between two nodes
    GreaterThan { left: NodeId, right: NodeId },

    /// An access operation between two nodes
    Access { left: NodeId, right: NodeId },

    /// An apply operation between two nodes
    Apply { left: NodeId, right: NodeId },

    /// A list append operation between two nodes
    ListAppend { left: NodeId, right: NodeId },

    /// A string concatination operation between two nodes
    StrConcat  { left: NodeId, right: NodeId },
}

impl Node {
    pub fn label(&self) -> String {
        match self {
            Node::Int { data } => format!("{data:#x} ({data})"),
            Node::Float { data } => format!("{data:.4}"),
            Node::Var { data } => format!("{data}"),
            Node::Bytes { data } => {
                format!("{:?}", data.iter().map(|x| *x as char).collect::<Vec<_>>())
            }
            Node::Sub { .. } => format!("-"),
            Node::Add { .. } => format!("+"),
            Node::Mul { .. } => format!("*"),
            Node::Exp { .. } => format!("^"),
            Node::GreaterThan { .. } => format!(">"),
            Node::Access { .. } => format!("@ (ACCESS) "),
            Node::Apply { .. } => format!("APPLY"),
            Node::ListAppend { .. } => format!("+< (LIST_APPEND)"),
            Node::StrConcat { .. } => format!("++ (STR_CONCAT)"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Token {
    id: TokenId,
    pos: u32,
}

impl Token {
    pub fn new(id: TokenId, pos: u32) -> Self {
        Self { id, pos }
    }
}

#[derive(Debug, PartialEq)]
pub enum TokenError {
    FloatWithMultipleDecimalPoints,
    QuoteInVariable,
    DotDot,
    UnquotedEndOfString,
    UnexpectedEndOfFile,
    InvalidSymbol,
}

#[derive(Debug)]
pub enum ParseError {
    NoTokensGiven,
    Int(std::num::ParseIntError),
    Float(std::num::ParseFloatError),
    Base85(base85::Error),
    Base64(base64::DecodeError),
}

/// A basic syntax tree of nodes
#[derive(Default, Debug, PartialEq)]
pub struct SyntaxTree {
    nodes: Vec<Node>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(usize);

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Index<NodeId> for Vec<Node> {
    type Output = Node;

    fn index(&self, index: NodeId) -> &Self::Output {
        &self[index.0]
    }
}

impl IndexMut<NodeId> for Vec<Node> {
    fn index_mut(&mut self, index: NodeId) -> &mut Self::Output {
        &mut self[index.0]
    }
}

/// Implement a node that has a single `data` child
macro_rules! impl_data_node {
    ($func_name:ident, $node:ident, $ty:ty) => {
        pub fn $func_name(&mut self, data: $ty) -> NodeId {
            let node_index = self.nodes.len();

            let node = Node::$node { data };
            self.nodes.push(node);

            NodeId(node_index)
        }
    };
}

/// Implement a node that has `left` and `right` children
macro_rules! impl_left_right_node {
    ($func_name:ident, $node:ident) => {
        pub fn $func_name(&mut self, left: NodeId, right: NodeId) -> NodeId {
            let node_index = self.nodes.len();

            let node = Node::$node { left, right };
            self.nodes.push(node);

            NodeId(node_index)
        }
    };
}

impl SyntaxTree {
    impl_data_node!(int, Int, i64);
    impl_data_node!(float, Float, f64);
    impl_data_node!(name, Var, String);
    impl_data_node!(bytes, Bytes, Vec<u8>);
    impl_left_right_node!(sub, Sub);
    impl_left_right_node!(add, Add);
    impl_left_right_node!(mul, Mul);
    impl_left_right_node!(greater_than, GreaterThan);
    impl_left_right_node!(access, Access);
    impl_left_right_node!(apply, Apply);
    impl_left_right_node!(exp, Exp);
    impl_left_right_node!(list_append, ListAppend);
    impl_left_right_node!(str_concat, StrConcat);

    /// Dump a .dot of this syntax tree
    pub fn dump_dot(&self, root: NodeId, out_name: &str) {
        let mut queue = vec![root];
        let mut seen_nodes = HashSet::new();

        let mut dot = String::from("digraph {\n");

        while let Some(node_id) = queue.pop() {
            // Ignore nodes we've already seen
            if !seen_nodes.insert(node_id) {
                continue;
            }

            let curr_node = &self.nodes[node_id];

            dot.push_str(&format!("{node_id} [ label = {:?} ];\n", curr_node.label()));

            match curr_node {
                Node::Sub { left, right }
                | Node::Add { left, right }
                | Node::Mul { left, right }
                | Node::GreaterThan { left, right }
                | Node::Exp { left, right }
                | Node::Apply { left, right }
                | Node::ListAppend { left, right }
                | Node::StrConcat { left, right }
                | Node::Access { left, right } => {
                    queue.push(*left);
                    queue.push(*right);

                    dot.push_str(&format!("{node_id} -> {left}  [ label=\"left\"; ];\n"));
                    dot.push_str(&format!("{node_id} -> {right} [ label=\"right\"; ];\n"));
                }


                Node::Int { .. } | Node::Float { .. } | Node::Var { .. } | Node::Bytes { .. } => {
                    // This is a leaf node.. nothing else to parse
                }
            }
        }

        dot.push('}');

        println!("{dot}");

        std::fs::write(out_name, dot).expect("Failed to write dot file");
    }
}

/// Convert the input program into a list of tokens
pub fn tokenize(input: &str) -> Result<Vec<Token>, (TokenError, usize)> {
    let mut result = Vec::new();
    let input: &str = input.into();
    let input = input.as_bytes();

    let mut index = 0;
    let mut in_token = false;

    let mut pos;

    'done: loop {
        if index >= input.len() {
            result.push(Token {
                id: TokenId::EndOfFile,
                pos: index as u32,
            });
            break 'done;
        }

        macro_rules! continue_while_space {
            () => {
                // Skip over all whitespace. Reset the current token when hitting whitespace.
                while matches!(input[index], b' ' | b'\n' | b'\r' | b'\t') {
                    index += 1;

                    if index >= input.len() {
                        result.push(Token {
                            id: TokenId::EndOfFile,
                            pos: input.len() as u32,
                        });
                        break 'done;
                    }

                    // Tags are split by spaces. If we see a space, reset the current token
                    in_token = false;
                }
            };
        }

        macro_rules! continue_while {
            ($until:pat) => {
                // Skip over all whitespace. Reset the current token when hitting whitespace.
                while matches!(input[index], $until) {
                    index += 1;

                    if index >= input.len() {
                        result.push(Token {
                            id: TokenId::EndOfFile,
                            pos: input.len() as u32,
                        });
                        break 'done;
                    }
                }
            };
        }

        macro_rules! continue_until {
            ($until:pat) => {
                // Skip over all whitespace. Reset the current token when hitting whitespace.
                while !matches!(input[index], $until) {
                    index += 1;

                    if index >= input.len() {
                        result.push(Token {
                            id: TokenId::EndOfFile,
                            pos: input.len() as u32,
                        });
                        break 'done;
                    }
                }

                // Skip over the found pattern
                index += 1;
            };
        }

        macro_rules! set_token {
            ($token:ident, $num_chars:literal) => {
                if !in_token {
                    result.push(Token {
                        id: TokenId::$token,
                        pos: pos as u32,
                    });
                }

                in_token = false;
                index += $num_chars;
                if index >= input.len() {
                    result.push(Token {
                        id: TokenId::EndOfFile,
                        pos: input.len() as u32,
                    });
                    break 'done;
                }
            };
        }

        // Continue past the whitespace
        continue_while_space!();
        pos = index;

        // println!("{pos}: {}", input[index] as char);
        match &input[index..] {
            [b'-', b'-', ..] => {
                // Comments are -- COMMENT
                set_token!(Comment, 2);
                continue_until!(b'\n');
            }
            [b'"', ..] => {
                set_token!(String, 1);

                // Skip over all whitespace. Reset the current token when hitting whitespace.
                while !matches!(input[index], b'"') {
                    index += 1;

                    if index >= input.len() {
                        return Err((TokenError::UnquotedEndOfString, pos));
                    }
                }

                // Skip over the found pattern
                index += 1;
            }
            [b'.', b'.', b'.', ..] => {
                set_token!(OpSpread, 3);
            }
            [b'.', b'.', ..] => {
                return Err((TokenError::DotDot, pos));
            }
            [b'.', b' ', ..] => {
                set_token!(OpWhere, 2);
            }
            [b'~', b'~', b'8', b'5', b'\'', ..] => {
                pos += 5;
                index += 5;
                set_token!(Base85, 5);

                continue_while!(b'0'..=b'9' | b'A'..=b'Z' | b'a'..=b'z' | 
                    b'!' | b'#' | b'$' | b'%' | b'&' | b'(' | b')' | b'*' | 
                    b'+' | b'-' | b';' | b'<' | b'=' | b'>' | b'?' | b'@' | 
                    b'^' | b'_' | b'`' | b'{' | b'|' | b'}' | b'~' | b'"');

                println!("Base85: After: {pos} {index}");
            }
            [b'~', b'~', b'3', b'2', b'\'', ..] => {
                pos += 5;
                index += 5;
                set_token!(Base32, 5);
                continue_while!(b'A'..=b'Z' | b'2'..=b'7' | b'=');
                println!("Base32: After: {pos} {index}");
            }
            [b'~', b'~', b'1', b'6', b'\'', ..] => {
                pos += 5;
                index += 5;
                set_token!(Base16, 5);
                continue_while!(b'A'..=b'F' | b'a'..=b'f' | b'0'..=b'9');
            }
            [b'~', b'~', b'6', b'4', b'\'', ..] => {
                pos += 5;
                index += 5;
                set_token!(Base64, 5);
                continue_while!(b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'=');
                println!("Base64: After: {pos} {index}");
            }
            [b'~', b'~', ..] => {
                pos += 2;
                index += 2;
                set_token!(Base64, 2);
                continue_while!(b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'=');
            }
            [b'0'..=b'9' | b'.', ..] => {
                let mut id = TokenId::Int;

                // Read all digits and potential '.' for floats
                while matches!(input[index], b'0'..=b'9' | b'.') {
                    if input[index] == b'.' {
                        if id == TokenId::Float {
                            return Err((TokenError::FloatWithMultipleDecimalPoints, index));
                        }

                        id = TokenId::Float;
                    };

                    index += 1;

                    if index >= input.len() {
                        result.push(Token::new(id, pos as u32));

                        result.push(Token {
                            id: TokenId::EndOfFile,
                            pos: index as u32,
                        });
                        break 'done;
                    }
                }

                result.push(Token::new(id, pos as u32));
                in_token = false;
            }

            // Name [a-zA-z$'_][a-zA-Z$'_0-9]*
            [b'a'..=b'z' | b'A'..=b'Z' | b'$' | b'\'' | b'_', ..] => {
                // Only allow quotes for names that start with $
                let allow_quotes = input[index] == b'$';

                set_token!(Name, 1);

                // Skip over all characters allowed in a name
                while matches!(input[index], b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'$' | b'_' | b'\'')
                {
                    // Explicitly disallow quotes in the variable
                    if !allow_quotes && input[index] == b'\'' {
                        return Err((TokenError::QuoteInVariable, index));
                    }

                    index += 1;

                    if index >= input.len() {
                        result.push(Token {
                            id: TokenId::EndOfFile,
                            pos: index as u32,
                        });
                        break 'done;
                    }
                }
            }
            [b'-', b'>', ..] => {
                set_token!(OpFunction, 2);
            }
            [b'>', b'>', ..] => {
                set_token!(OpCompose, 2);
            }
            [b'<', b'<', ..] => {
                set_token!(OpComposeReverse, 2);
            }
            [b'=', b'=', ..] => {
                set_token!(OpEqual, 2);
            }
            [b'/', b'/', ..] => {
                set_token!(OpFloorDiv, 2);
            }
            [b'/', b'=', ..] => {
                set_token!(OpNotEqual, 2);
            }
            [b'<', b'=', ..] => {
                set_token!(OpLessEqual, 2);
            }
            [b'>', b'=', ..] => {
                set_token!(OpGreaterEqual, 2);
            }
            [b'&', b'&', ..] => {
                set_token!(OpBoolAnd, 2);
            }
            [b'|', b'|', ..] => {
                set_token!(OpBoolOr, 2);
            }
            [b'+', b'+', ..] => {
                set_token!(OpStrConcat, 2);
            }
            [b'>', b'+', ..] => {
                set_token!(OpListCons, 2);
            }
            [b'+', b'<', ..] => {
                set_token!(OpListAppend, 2);
            }
            [b'|', b'>', ..] => {
                set_token!(OpPipe, 2);
            }
            [b'<', b'|', ..] => {
                set_token!(OpReversePipe, 2);
            }
            [b'#', ..] => {
                index += 1;

                let mut is_empty = true;
                continue_while_space!();

                pos = index;
                set_token!(Symbol, 0);

                // Skip over all characters allowed in a name
                while matches!(input[index], b'a'..=b'z' | b'A'..=b'Z' | b'$' | b'_' | b'\'') {
                    index += 1;
                    is_empty = false;

                    if index >= input.len() {
                        result.push(Token {
                            id: TokenId::EndOfFile,
                            pos: index as u32,
                        });

                        break 'done;
                    }
                }

                if is_empty {
                    if index >= input.len() {
                        return Err((TokenError::UnexpectedEndOfFile, pos));
                    } else {
                        return Err((TokenError::InvalidSymbol, pos));
                    }
                }
            }
            [b'*', ..] => {
                set_token!(OpMul, 1);
            }
            [b'+', ..] => {
                set_token!(OpAdd, 1);
            }
            [b'/', ..] => {
                set_token!(OpDiv, 1);
            }
            [b'^', ..] => {
                set_token!(OpExp, 1);
            }
            [b'!', ..] => {
                set_token!(OpRightEval, 1);
            }
            [b'@', ..] => {
                set_token!(OpAccess, 1);
            }
            [b':', ..] => {
                set_token!(OpHasType, 1);
            }
            [b'%', ..] => {
                set_token!(OpMod, 1);
            }
            [b'<', ..] => {
                set_token!(OpLess, 1);
            }
            [b'>', ..] => {
                set_token!(OpGreater, 1);
            }
            [b'=', ..] => {
                set_token!(OpAssign, 1);
            }
            [b'|', ..] => {
                set_token!(OpMatchCase, 1);
            }
            [b'(', ..] => {
                set_token!(LeftParen, 1);
            }
            [b')', ..] => {
                set_token!(RightParen, 1);
            }
            [b'{', ..] => {
                set_token!(LeftBracket, 1);
            }
            [b'}', ..] => {
                set_token!(RightBracket, 1);
            }
            [b',', ..] => {
                set_token!(Comma, 1);
            }
            [b'[', ..] => {
                set_token!(LeftBrace, 1);
            }
            [b']', ..] => {
                set_token!(RightBrace, 1);
            }
            [b'?', ..] => {
                set_token!(OpAssert, 1);
            }
            [b'-', ..] => {
                set_token!(OpSub, 1);

                // Reset the token for negative numbers -123 vs subtract of a - 123
                if matches!(input[index], b'0'..=b'9') {
                    in_token = false;
                }
            }
            _ => {
                panic!("Unknown {}", input[index] as char);
            }
        }
    }

    Ok(result)
}

pub fn parse(
    tokens: &[Token],
    token_index: &mut usize,
    input: &str,
    ast: &mut SyntaxTree,
    current_precedence: u32,
) -> Result<NodeId, (ParseError, usize)> {
    if tokens.is_empty() {
        return Err((ParseError::NoTokensGiven, 0));
    }

    let input_bytes = input.as_bytes();

    let parse_leaf = |ast: &mut SyntaxTree,
                      token_index: &mut usize|
     -> Result<NodeId, (ParseError, usize)> {
        let start_token_pos = *token_index;

        let start = tokens[start_token_pos];
        *token_index += 1;

        let input_index = start.pos as usize;
        let mut end_input_index = tokens[*token_index].pos as usize - 1;

        dbg!(start.id);

        macro_rules! continue_while {
            ($pat:pat) => {
                // Skip over all whitespace. Reset the current token when hitting whitespace.
                while end_input_index > input_index && !matches!(input_bytes[end_input_index], $pat)
                {
                    end_input_index -= 1;
                }
            };
        }

        // Parse the next leaf token
        let leaf = match start.id {
            TokenId::Int => {
                continue_while!(b'0'..=b'9');

                let value = input[input_index..=end_input_index]
                    .parse()
                    .map_err(|e| (ParseError::Int(e), input_index))?;

                // Add an Int node
                ast.int(value)
            }
            TokenId::Float => {
                continue_while!(b'0'..=b'9' | b'.');

                let value = input[input_index..=end_input_index]
                    .parse()
                    .map_err(|e| (ParseError::Float(e), input_index))?;

                // Add an Int node
                ast.float(value)
            }
            TokenId::OpSub => {
                let zero_int = ast.int(0);

                let right = parse(tokens, token_index, input, ast, 5000)?;

                ast.sub(zero_int, right)
            }
            TokenId::Name => {
                continue_while!(b'a'..=b'z' | b'A'..=b'Z' | b'$' | b'\'' | b'_' | b'0'..=b'9');
                let name = input[input_index..=end_input_index].to_string();
                ast.name(name)
            }
            TokenId::OpAdd => {
                panic!("Hit Add as a left operation!?")
            }

            TokenId::Base85 => {
                continue_while!(b'0'..=b'9' | b'A'..=b'Z' | b'a'..=b'z' | 
                    b'!' | b'#' | b'$' | b'%' | b'&' | b'(' | b')' | b'*' | 
                    b'+' | b'-' | b';' | b'<' | b'=' | b'>' | b'?' | b'@' | 
                    b'^' | b'_' | b'`' | b'{' | b'|' | b'}' | b'~' | b'"');
                let name = &input[input_index..=end_input_index];

                let bytes = base85::decode(name)
                    .map_err(|e| (ParseError::Base85(e), start.pos as usize))?;
                ast.bytes(bytes)
            }
            TokenId::Base64 => {
                use base64::Engine;

                let name = &input[input_index..=end_input_index];
                continue_while!(b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'=');

                // Decode the base64 string
                let bytes = base64::engine::general_purpose::STANDARD
                    .decode(name)
                    .map_err(|e| (ParseError::Base64(e), start.pos as usize))?;

                // Create the decoded bytes string
                ast.bytes(bytes)
            }
            x => panic!("Unknown parse token: {x:?}"),
        };

        Ok(leaf)
    };

    let mut left = parse_leaf(ast, token_index)?;
    println!("Left: {:?}", ast.nodes[left]);

    //
    loop {
        println!("{input}");
        println!("{}^", " ".repeat(*token_index));

        // Check if the left was modified
        let original = left;

        let Some(next) = tokens.get(*token_index) else {
            println!("Breaking out of the loop..");
            break;
        };

        if matches!(next.id, TokenId::EndOfFile) {
            break;
        }

        if let Some(binary_prescedence) = next.id.get_op_prescedence() {
            let binary_op = next;
            // Is a binary operator

            // If the next prescedence is less than the current, return out of the loop
            if binary_prescedence <= current_precedence {
                println!("Found smaller.. bailing");
                return Ok(left);
            }

            // Increment the token index
            *token_index += 1;

            let right = parse(tokens, token_index, input, ast, binary_prescedence)?;
            match binary_op.id {
                TokenId::EndOfFile => {
                    println!("Hit EOF");
                    break;
                }

                TokenId::OpAdd => {
                    left = ast.add(left, right);
                }

                TokenId::OpSub => {
                    left = ast.sub(left, right);
                }

                TokenId::OpMul => {
                    left = ast.mul(left, right);
                }

                TokenId::OpExp => {
                    left = ast.exp(left, right);
                }

                TokenId::OpGreater => {
                    left = ast.greater_than(left, right);
                }

                TokenId::OpAccess => {
                    left = ast.access(left, right);
                }

                TokenId::OpListAppend => {
                    left = ast.list_append(left, right);
                }

                TokenId::OpStrConcat => {
                    left = ast.str_concat(left, right);
                }

                x => panic!("Unknown operator token: {x:?}"),
            }
        } else {
            println!("Not a bin op: {next:?}");

            if TokenId::OpAccess.get_op_prescedence().unwrap() < current_precedence {
                break;
            }

            let right = parse(tokens, token_index, input, ast, current_precedence + 1)?;

            left = ast.apply(left, right);
        }

        // Base case: The tree didn't move, we've reached the end
        if left == original {
            break;
        }
    }

    println!("Out of loop..");
    Ok(left)
}

#[cfg(test)]
mod tests {
    use super::*;
    use TokenId::*;

    #[test]
    fn test_tokenize_digit() {
        let tokens = tokenize("1");
        assert_eq!(
            tokens,
            Ok(vec![Token::new(Int, 0), Token::new(EndOfFile, 1)])
        );
    }

    #[test]
    fn test_tokenize_multiple_digits() {
        let tokens = tokenize("123");
        assert_eq!(
            tokens,
            Ok(vec![Token::new(Int, 0), Token::new(EndOfFile, 3)])
        );
    }

    #[test]
    fn test_tokenize_negative_int() {
        let tokens = tokenize("-123");
        assert_eq!(
            tokens,
            Ok(vec![
                Token::new(OpSub, 0),
                Token::new(Int, 1),
                Token::new(EndOfFile, 4)
            ])
        );
    }

    #[test]
    fn test_tokenize_float() {
        let tokens = tokenize("3.14");
        assert_eq!(
            tokens,
            Ok(vec![Token::new(Float, 0), Token::new(EndOfFile, 4)])
        );
    }

    #[test]
    fn test_tokenize_negative_float() {
        let tokens = tokenize("-3.14");
        assert_eq!(
            tokens,
            Ok(vec![
                Token::new(OpSub, 0),
                Token::new(Float, 1),
                Token::new(EndOfFile, 5)
            ])
        );
    }

    #[test]
    fn test_tokenisze_float_no_int_part() {
        let tokens = tokenize(".14");
        assert_eq!(
            tokens,
            Ok(vec![Token::new(Float, 0), Token::new(EndOfFile, 3)])
        );
    }

    #[test]
    fn test_tokenize_float_no_decimal_part() {
        let tokens = tokenize("10.");
        assert_eq!(
            tokens,
            Ok(vec![Token::new(Float, 0), Token::new(EndOfFile, 3)])
        );
    }

    #[test]
    fn test_tokenize_float_with_multiple_decimal_points() {
        let tokens = tokenize("10.0.0");
        assert_eq!(tokens, Err((TokenError::FloatWithMultipleDecimalPoints, 4)));
    }

    #[test]
    fn test_tokensize_binop() {
        let tokens = tokenize("1 + 2");
        assert_eq!(
            tokens,
            Ok(vec![
                Token::new(Int, 0),
                Token::new(OpAdd, 2),
                Token::new(Int, 4),
                Token::new(EndOfFile, 5),
            ])
        );
    }

    #[test]
    fn test_tokensize_binop_no_spaces() {
        let tokens = tokenize("1+2");
        assert_eq!(
            tokens,
            Ok(vec![
                Token::new(Int, 0),
                Token::new(OpAdd, 1),
                Token::new(Int, 2),
                Token::new(EndOfFile, 3),
            ])
        );
    }

    #[test]
    fn test_tokensize_binop_var() {
        for (op, ident) in [
            ("+", OpAdd),
            ("-", OpSub),
            ("*", OpMul),
            ("/", OpDiv),
            ("^", OpExp),
            ("%", OpMod),
            ("==", OpEqual),
            ("//", OpFloorDiv),
            ("/=", OpNotEqual),
            ("<", OpLess),
            (">", OpGreater),
            ("<=", OpLessEqual),
            (">=", OpGreaterEqual),
            ("&&", OpBoolAnd),
            ("||", OpBoolOr),
            ("++", OpStrConcat),
            (">+", OpListCons),
            ("+<", OpListAppend),
            ("!", OpRightEval),
            (":", OpHasType),
            ("|>", OpPipe),
            ("<|", OpReversePipe),
        ] {
            let input = format!("a {op} b");
            let tokens = tokenize(&input);
            assert_eq!(
                tokens,
                Ok(vec![
                    Token::new(Name, 0),
                    Token::new(ident, 2),
                    Token::new(Name, 3 + op.len() as u32),
                    Token::new(EndOfFile, 4 + op.len() as u32),
                ])
            );

            let input = format!("aaa{op}bbb");
            let tokens = tokenize(&input);
            assert_eq!(
                tokens,
                Ok(vec![
                    Token::new(Name, 0),
                    Token::new(ident, 3),
                    Token::new(Name, 3 + op.len() as u32),
                    Token::new(EndOfFile, 6 + op.len() as u32),
                ])
            );
        }
    }

    #[test]
    fn test_tokenize_var() {
        let tokens = tokenize("abc");
        assert_eq!(
            tokens,
            Ok(vec![Token::new(Name, 0), Token::new(EndOfFile, 3),])
        );
    }

    #[test]
    fn test_tokenize_var_with_quote() {
        let tokens = tokenize("sha1'abc");
        assert_eq!(tokens, Err((TokenError::QuoteInVariable, 4)));
    }

    #[test]
    fn test_tokenize_dollar_sha1_var() {
        let tokens = tokenize("$sha1'abc");
        assert_eq!(
            tokens,
            Ok(vec![Token::new(Name, 0), Token::new(EndOfFile, 9)])
        );
    }

    #[test]
    fn test_tokenize_dollar_dollar_var() {
        let tokens = tokenize("$$bills");
        assert_eq!(
            tokens,
            Ok(vec![Token::new(Name, 0), Token::new(EndOfFile, 7)])
        );
    }

    #[test]
    fn test_tokenize_dot_dot_raises_parse_error() {
        let tokens = tokenize("..");
        assert_eq!(tokens, Err((TokenError::DotDot, 0)));
    }

    #[test]
    fn test_tokenize_spread() {
        let tokens = tokenize("...");
        assert_eq!(
            tokens,
            Ok(vec![Token::new(OpSpread, 0), Token::new(EndOfFile, 3)])
        );
    }

    #[test]
    fn test_ignore_whitespace() {
        let tokens = tokenize("1\n+\t2");
        assert_eq!(
            tokens,
            Ok(vec![
                Token::new(Int, 0),
                Token::new(OpAdd, 2),
                Token::new(Int, 4),
                Token::new(EndOfFile, 5)
            ])
        );
    }

    #[test]
    fn test_ignore_line_comment() {
        let tokens = tokenize("-- 1\n2");
        assert_eq!(
            tokens,
            Ok(vec![
                Token::new(Comment, 0),
                Token::new(Int, 5),
                Token::new(EndOfFile, 6)
            ])
        );
    }

    #[test]
    fn test_tokenize_string() {
        let tokens = tokenize("\"hello\"");
        assert_eq!(
            tokens,
            Ok(vec![Token::new(String, 0), Token::new(EndOfFile, 7)])
        );
    }

    #[test]
    fn test_tokenize_string_with_spaces() {
        let tokens = tokenize("\"hello world\"");
        assert_eq!(
            tokens,
            Ok(vec![Token::new(String, 0), Token::new(EndOfFile, 13)])
        );
    }

    #[test]
    fn test_tokenize_string_missing_end_quote() {
        let tokens = tokenize("\"hello world");
        assert_eq!(tokens, Err((TokenError::UnquotedEndOfString, 0)));
    }

    #[test]
    fn test_tokenisze_empty_list() {
        let tokens = tokenize("[ ]");
        assert_eq!(
            tokens,
            Ok(vec![
                Token::new(LeftBrace, 0),
                Token::new(RightBrace, 2),
                Token::new(EndOfFile, 3)
            ])
        );
    }

    #[test]
    fn test_tokenize_list_with_items() {
        let tokens = tokenize("[ 1 , 2 ]");

        assert_eq!(
            tokens,
            Ok(vec![
                Token::new(LeftBrace, 0),
                Token::new(Int, 2),
                Token::new(Comma, 4),
                Token::new(Int, 6),
                Token::new(RightBrace, 8),
                Token::new(EndOfFile, 9)
            ])
        );
    }

    #[test]
    fn test_tokenize_list_with_items_with_no_spaces() {
        let tokens = tokenize("[1,2]");

        assert_eq!(
            tokens,
            Ok(vec![
                Token::new(LeftBrace, 0),
                Token::new(Int, 1),
                Token::new(Comma, 2),
                Token::new(Int, 3),
                Token::new(RightBrace, 4),
                Token::new(EndOfFile, 5)
            ])
        );
    }
    #[test]

    fn test_tokenize_function() {
        let tokens = tokenize("a -> b -> a + b");

        assert_eq!(
            tokens,
            Ok(vec![
                Token::new(Name, 0),
                Token::new(OpFunction, 2),
                Token::new(Name, 5),
                Token::new(OpFunction, 7),
                Token::new(Name, 10),
                Token::new(OpAdd, 12),
                Token::new(Name, 14),
                Token::new(EndOfFile, 15)
            ])
        );
    }

    #[test]
    fn test_tokenize_function_no_spaces() {
        let tokens = tokenize("a->b->a+b");

        assert_eq!(
            tokens,
            Ok(vec![
                Token::new(Name, 0),
                Token::new(OpFunction, 1),
                Token::new(Name, 3),
                Token::new(OpFunction, 4),
                Token::new(Name, 6),
                Token::new(OpAdd, 7),
                Token::new(Name, 8),
                Token::new(EndOfFile, 9)
            ])
        );
    }

    #[test]
    fn test_tokenize_where() {
        let tokens = tokenize("a . b");

        assert_eq!(
            tokens,
            Ok(vec![
                Token::new(Name, 0),
                Token::new(OpWhere, 2),
                Token::new(Name, 4),
                Token::new(EndOfFile, 5)
            ])
        );
    }

    #[test]
    fn test_tokenize_assert() {
        let tokens = tokenize("a ? b");

        assert_eq!(
            tokens,
            Ok(vec![
                Token::new(Name, 0),
                Token::new(OpAssert, 2),
                Token::new(Name, 4),
                Token::new(EndOfFile, 5)
            ])
        );
    }

    #[test]
    fn test_tokenize_hastype() {
        let tokens = tokenize("a : b");

        assert_eq!(
            tokens,
            Ok(vec![
                Token::new(Name, 0),
                Token::new(OpHasType, 2),
                Token::new(Name, 4),
                Token::new(EndOfFile, 5)
            ])
        );
    }

    #[test]
    fn test_tokenize_bytes_return_bytes_base64() {
        let tokens = tokenize("~~QUJD");

        assert_eq!(
            tokens,
            Ok(vec![Token::new(Base64, 2), Token::new(EndOfFile, 6)])
        );
    }

    #[test]
    fn test_tokenize_base85() {
        let tokens = tokenize("~~85'K|(_");

        assert_eq!(
            tokens,
            Ok(vec![Token::new(Base85, 5), Token::new(EndOfFile, 9)])
        );
    }

    #[test]
    fn test_tokenize_base64() {
        let tokens = tokenize("~~64'K|(_");

        assert_eq!(
            tokens,
            Ok(vec![Token::new(Base64, 5), Token::new(EndOfFile, 9)])
        );
    }

    #[test]
    fn test_tokenize_base32() {
        let tokens = tokenize("~~32'K|(_");

        assert_eq!(
            tokens,
            Ok(vec![Token::new(Base32, 5), Token::new(EndOfFile, 9)])
        );
    }

    #[test]
    fn test_tokenize_hole() {
        let tokens = tokenize("()");

        assert_eq!(
            tokens,
            Ok(vec![
                Token::new(LeftParen, 0),
                Token::new(RightParen, 1),
                Token::new(EndOfFile, 2)
            ])
        );
    }

    #[test]
    fn test_tokenize_hole_with_spaces() {
        let tokens = tokenize("(  )");

        assert_eq!(
            tokens,
            Ok(vec![
                Token::new(LeftParen, 0),
                Token::new(RightParen, 3),
                Token::new(EndOfFile, 4)
            ])
        );
    }

    #[test]
    fn test_token_parenthetical_expression() {
        let tokens = tokenize("(1+2)");

        assert_eq!(
            tokens,
            Ok(vec![
                Token::new(LeftParen, 0),
                Token::new(Int, 1),
                Token::new(OpAdd, 2),
                Token::new(Int, 3),
                Token::new(RightParen, 4),
                Token::new(EndOfFile, 5)
            ])
        );
    }

    #[test]
    fn test_tokenize_pipe() {
        let tokens = tokenize("1 |> f . f = a -> a + 1");

        assert_eq!(
            tokens,
            Ok(vec![
                Token::new(Int, 0),
                Token::new(OpPipe, 2),
                Token::new(Name, 5),
                Token::new(OpWhere, 7),
                Token::new(Name, 9),
                Token::new(OpAssign, 11),
                Token::new(Name, 13),
                Token::new(OpFunction, 15),
                Token::new(Name, 18),
                Token::new(OpAdd, 20),
                Token::new(Int, 22),
                Token::new(EndOfFile, 23)
            ])
        );
    }

    #[test]
    fn test_tokenize_reverse_pipe() {
        let tokens = tokenize("1 <| 1 . f = a -> a + 1");

        assert_eq!(
            tokens,
            Ok(vec![
                Token::new(Int, 0),
                Token::new(OpReversePipe, 2),
                Token::new(Int, 5),
                Token::new(OpWhere, 7),
                Token::new(Name, 9),
                Token::new(OpAssign, 11),
                Token::new(Name, 13),
                Token::new(OpFunction, 15),
                Token::new(Name, 18),
                Token::new(OpAdd, 20),
                Token::new(Int, 22),
                Token::new(EndOfFile, 23)
            ])
        );
    }

    #[test]
    fn test_tokenize_record_no_fields_no_spaces() {
        let tokens = tokenize("{}");

        assert_eq!(
            tokens,
            Ok(vec![
                Token::new(LeftBracket, 0),
                Token::new(RightBracket, 1),
                Token::new(EndOfFile, 2)
            ])
        );
    }

    #[test]
    fn test_tokenize_record_no_fields() {
        let tokens = tokenize("{  }");

        assert_eq!(
            tokens,
            Ok(vec![
                Token::new(LeftBracket, 0),
                Token::new(RightBracket, 3),
                Token::new(EndOfFile, 4)
            ])
        );
    }

    #[test]
    fn test_tokenize_record_one_field() {
        let tokens = tokenize("{ a = 4 }");

        assert_eq!(
            tokens,
            Ok(vec![
                Token::new(LeftBracket, 0),
                Token::new(Name, 2),
                Token::new(OpAssign, 4),
                Token::new(Int, 6),
                Token::new(RightBracket, 8),
                Token::new(EndOfFile, 9)
            ])
        );
    }

    #[test]
    fn test_tokenize_record_multiple_fields() {
        let tokens = tokenize(r#"{ a = 4, b = "z" }"#);

        assert_eq!(
            tokens,
            Ok(vec![
                Token::new(LeftBracket, 0),
                Token::new(Name, 2),
                Token::new(OpAssign, 4),
                Token::new(Int, 6),
                Token::new(Comma, 7),
                Token::new(Name, 9),
                Token::new(OpAssign, 11),
                Token::new(String, 13),
                Token::new(RightBracket, 17),
                Token::new(EndOfFile, 18)
            ])
        );
    }

    #[test]
    fn test_tokenize_record_access() {
        let tokens = tokenize("r@a");

        assert_eq!(
            tokens,
            Ok(vec![
                Token::new(Name, 0),
                Token::new(OpAccess, 1),
                Token::new(Name, 2),
                Token::new(EndOfFile, 3)
            ])
        );
    }

    #[test]
    fn test_tokenize_right_eval() {
        let tokens = tokenize("a!b");

        assert_eq!(
            tokens,
            Ok(vec![
                Token::new(Name, 0),
                Token::new(OpRightEval, 1),
                Token::new(Name, 2),
                Token::new(EndOfFile, 3)
            ])
        );
    }

    #[test]
    fn test_tokenize_match() {
        let tokens = tokenize("g = | 1 -> 2 | 2 -> 3");

        assert_eq!(
            tokens,
            Ok(vec![
                Token::new(Name, 0),
                Token::new(OpAssign, 2),
                Token::new(OpMatchCase, 4),
                Token::new(Int, 6),
                Token::new(OpFunction, 8),
                Token::new(Int, 11),
                Token::new(OpMatchCase, 13),
                Token::new(Int, 15),
                Token::new(OpFunction, 17),
                Token::new(Int, 20),
                Token::new(EndOfFile, 21)
            ])
        );
    }

    #[test]
    fn test_tokenize_compose() {
        let tokens = tokenize("f >> g");

        assert_eq!(
            tokens,
            Ok(vec![
                Token::new(Name, 0),
                Token::new(OpCompose, 2),
                Token::new(Name, 5),
                Token::new(EndOfFile, 6)
            ])
        );
    }

    #[test]
    fn test_tokenize_compose_reverse() {
        let tokens = tokenize("f << g");

        assert_eq!(
            tokens,
            Ok(vec![
                Token::new(Name, 0),
                Token::new(OpComposeReverse, 2),
                Token::new(Name, 5),
                Token::new(EndOfFile, 6)
            ])
        );
    }

    #[test]
    fn test_tokenize_list_with_only_spread() {
        let tokens = tokenize("[ ... ]");

        assert_eq!(
            tokens,
            Ok(vec![
                Token::new(LeftBrace, 0),
                Token::new(OpSpread, 2),
                Token::new(RightBrace, 6),
                Token::new(EndOfFile, 7)
            ])
        );
    }

    #[test]
    fn test_tokenize_list_with_spread() {
        let tokens = tokenize("[ 1 , ... ]");

        assert_eq!(
            tokens,
            Ok(vec![
                Token::new(LeftBrace, 0),
                Token::new(Int, 2),
                Token::new(Comma, 4),
                Token::new(OpSpread, 6),
                Token::new(RightBrace, 10),
                Token::new(EndOfFile, 11)
            ])
        );
    }

    #[test]
    fn test_tokenize_list_with_spread_no_spaces() {
        let tokens = tokenize("[1,...]");

        assert_eq!(
            tokens,
            Ok(vec![
                Token::new(LeftBrace, 0),
                Token::new(Int, 1),
                Token::new(Comma, 2),
                Token::new(OpSpread, 3),
                Token::new(RightBrace, 6),
                Token::new(EndOfFile, 7)
            ])
        );
    }

    #[test]
    fn test_tokenize_list_with_named_spread() {
        let tokens = tokenize("[1,...rest]");

        assert_eq!(
            tokens,
            Ok(vec![
                Token::new(LeftBrace, 0),
                Token::new(Int, 1),
                Token::new(Comma, 2),
                Token::new(OpSpread, 3),
                Token::new(Name, 6),
                Token::new(RightBrace, 10),
                Token::new(EndOfFile, 11)
            ])
        );
    }

    #[test]
    fn test_tokenize_record_with_only_spread() {
        let tokens = tokenize("{ ... }");

        assert_eq!(
            tokens,
            Ok(vec![
                Token::new(LeftBracket, 0),
                Token::new(OpSpread, 2),
                Token::new(RightBracket, 6),
                Token::new(EndOfFile, 7)
            ])
        );
    }

    #[test]
    fn test_tokenize_record_with_spread() {
        let tokens = tokenize("{ x = 1, ... }");

        assert_eq!(
            tokens,
            Ok(vec![
                Token::new(LeftBracket, 0),
                Token::new(Name, 2),
                Token::new(OpAssign, 4),
                Token::new(Int, 6),
                Token::new(Comma, 7),
                Token::new(OpSpread, 9),
                Token::new(RightBracket, 13),
                Token::new(EndOfFile, 14)
            ])
        );
    }

    #[test]
    fn test_tokenize_record_with_spread_no_spaces() {
        let tokens = tokenize("{x=1,...}");

        assert_eq!(
            tokens,
            Ok(vec![
                Token::new(LeftBracket, 0),
                Token::new(Name, 1),
                Token::new(OpAssign, 2),
                Token::new(Int, 3),
                Token::new(Comma, 4),
                Token::new(OpSpread, 5),
                Token::new(RightBracket, 8),
                Token::new(EndOfFile, 9)
            ])
        );
    }

    #[test]
    fn test_tokenize_symbol_with_space() {
        let tokens = tokenize("#   abc");

        assert_eq!(
            tokens,
            Ok(vec![Token::new(Symbol, 4), Token::new(EndOfFile, 7)])
        );
    }

    #[test]
    fn test_tokenize_symbol_with_no_space() {
        let tokens = tokenize("#abc");

        assert_eq!(
            tokens,
            Ok(vec![Token::new(Symbol, 1), Token::new(EndOfFile, 4)])
        );
    }

    #[test]
    fn test_tokenize_symbol_non_name_raises_parse_error() {
        let tokens = tokenize("#111");

        assert_eq!(tokens, Err((TokenError::InvalidSymbol, 1)));
    }

    #[test]
    fn test_parse_empty_tokens() {
        let mut ast = SyntaxTree::default();
        let mut token_position = 0;
        let tokens = parse(&[], &mut token_position, "", &mut ast, u32::MIN);

        assert!(tokens.is_err());
        assert!(matches!(tokens.err(), Some((ParseError::NoTokensGiven, 0))));
    }

    #[test]
    fn test_parse_digit_returns_int() {
        let mut ast = SyntaxTree::default();
        let mut token_position = 0;
        let input = "1";
        let tokens = tokenize(input).unwrap();
        parse(&tokens, &mut token_position, input, &mut ast, u32::MIN).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![Node::Int { data: 1 }]
            }
        );
    }

    #[test]
    fn test_parse_digits_returns_int() {
        let mut ast = SyntaxTree::default();
        let mut token_position = 0;

        let input = "123";
        let tokens = tokenize(input).unwrap();
        parse(&tokens, &mut token_position, input, &mut ast, u32::MIN).unwrap();
        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![Node::Int { data: 123 }]
            }
        );
    }

    #[test]
    fn test_parse_negative_int_returns_binary_sub_int() {
        let mut ast = SyntaxTree::default();
        let mut token_position = 0;
        let input = "-123";
        let tokens = tokenize(input).unwrap();
        let _root = parse(&tokens, &mut token_position, input, &mut ast, u32::MIN).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Int { data: 0 },
                    Node::Int { data: 123 },
                    Node::Sub {
                        left: NodeId(0),
                        right: NodeId(1)
                    }
                ]
            }
        );
    }

    #[test]
    fn test_parse_negative_var() {
        let mut ast = SyntaxTree::default();
        let mut token_position = 0;
        let input = "-x";
        let tokens = tokenize(input).unwrap();
        let _root = parse(&tokens, &mut token_position, input, &mut ast, u32::MIN).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Int { data: 0 },
                    Node::Var {
                        data: "x".to_string()
                    },
                    Node::Sub {
                        left: NodeId(0),
                        right: NodeId(1)
                    }
                ]
            }
        );
    }

    #[test]
    fn test_parse_negative_int_binds_tighter_than_plus() {
        let input = "-l+r";
        let mut token_position = 0;
        let mut ast = SyntaxTree::default();
        let tokens = tokenize(input).unwrap();
        let _root = parse(&tokens, &mut token_position, input, &mut ast, u32::MIN).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Int { data: 0 },
                    Node::Var {
                        data: "l".to_string()
                    },
                    Node::Sub {
                        left: NodeId(0),
                        right: NodeId(1)
                    },
                    Node::Var {
                        data: "r".to_string()
                    },
                    Node::Add {
                        left: NodeId(2),
                        right: NodeId(3)
                    },
                ]
            }
        );
    }

    #[test]
    fn test_parse_negative_int_binds_tighter_than_mul() {
        let input = "-l*r";
        let mut token_position = 0;
        let mut ast = SyntaxTree::default();
        let tokens = tokenize(input).unwrap();
        let _root = parse(&tokens, &mut token_position, input, &mut ast, u32::MIN).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Int { data: 0 },
                    Node::Var {
                        data: "l".to_string()
                    },
                    Node::Sub {
                        left: NodeId(0),
                        right: NodeId(1)
                    },
                    Node::Var {
                        data: "r".to_string()
                    },
                    Node::Mul {
                        left: NodeId(2),
                        right: NodeId(3)
                    },
                ]
            }
        );
    }

    #[test]
    fn test_parse_negative_int_binds_tighter_than_access() {
        let input = "-l@r";
        let mut token_position = 0;
        let mut ast = SyntaxTree::default();
        let tokens = tokenize(input).unwrap();
        let _root = parse(&tokens, &mut token_position, input, &mut ast, u32::MIN).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Int { data: 0 },
                    Node::Var {
                        data: "l".to_string()
                    },
                    Node::Sub {
                        left: NodeId(0),
                        right: NodeId(1)
                    },
                    Node::Var {
                        data: "r".to_string()
                    },
                    Node::Access {
                        left: NodeId(2),
                        right: NodeId(3)
                    },
                ]
            }
        );
    }

    #[test]
    fn test_parse_negative_int_binds_tighter_than_apply() {
        let input = "-l r";
        let mut token_position = 0;
        let mut ast = SyntaxTree::default();
        let tokens = tokenize(input).unwrap();

        let _root = parse(&tokens, &mut token_position, input, &mut ast, u32::MIN).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Int { data: 0 },
                    Node::Var {
                        data: "l".to_string()
                    },
                    Node::Sub {
                        left: NodeId(0),
                        right: NodeId(1)
                    },
                    Node::Var {
                        data: "r".to_string()
                    },
                    Node::Apply {
                        left: NodeId(2),
                        right: NodeId(3)
                    },
                ]
            }
        );
    }

    #[test]
    fn test_parse_decimal_return_float() {
        let input = "3.14";
        let mut token_position = 0;
        let mut ast = SyntaxTree::default();
        let tokens = tokenize(input).unwrap();

        let _root = parse(&tokens, &mut token_position, input, &mut ast, u32::MIN).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![Node::Float { data: 3.14 },]
            }
        );
    }
    #[test]
    fn test_parse_decimal_return_returns_binary_sub_float() {
        let input = "-3.14";
        let mut token_position = 0;
        let mut ast = SyntaxTree::default();
        let tokens = tokenize(input).unwrap();

        let _root = parse(&tokens, &mut token_position, input, &mut ast, u32::MIN).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Int { data: 0 },
                    Node::Float { data: 3.14 },
                    Node::Sub {
                        left: NodeId(0),
                        right: NodeId(1)
                    }
                ]
            }
        );
    }
    #[test]
    fn test_parse_var_returns_var() {
        let input = "abc_123";
        let mut token_position = 0;
        let mut ast = SyntaxTree::default();
        let tokens = tokenize(input).unwrap();

        let _root = parse(&tokens, &mut token_position, input, &mut ast, u32::MIN).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![Node::Var {
                    data: "abc_123".to_string()
                },]
            }
        );
    }

    #[test]
    fn test_parse_sha_var_returns_var() {
        let input = "$sha1'abc";
        let mut token_position = 0;
        let mut ast = SyntaxTree::default();
        let tokens = tokenize(input).unwrap();

        let _root = parse(&tokens, &mut token_position, input, &mut ast, u32::MIN).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![Node::Var {
                    data: "$sha1'abc".to_string()
                },]
            }
        );
    }

    #[test]
    fn test_parse_sha_var_quote_returns_var() {
        let input = "$sha1'abc";
        let mut token_position = 0;
        let mut ast = SyntaxTree::default();
        let tokens = tokenize(input).unwrap();

        let _root = parse(&tokens, &mut token_position, input, &mut ast, u32::MIN).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![Node::Var {
                    data: "$sha1'abc".to_string()
                },]
            }
        );
    }

    #[test]
    fn test_parse_dollar_dollar_returns_var() {
        let input = "$$";
        let mut token_position = 0;
        let mut ast = SyntaxTree::default();
        let tokens = tokenize(input).unwrap();

        let _root = parse(&tokens, &mut token_position, input, &mut ast, u32::MIN).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![Node::Var {
                    data: "$$".to_string()
                },]
            }
        );
    }

    #[test]
    fn test_parse_dollar_returns_var() {
        let input = "$";
        let mut token_position = 0;
        let mut ast = SyntaxTree::default();
        let tokens = tokenize(input).unwrap();

        let _root = parse(&tokens, &mut token_position, input, &mut ast, u32::MIN).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![Node::Var {
                    data: "$".to_string()
                },]
            }
        );
    }
    #[test]
    fn test_parse_dollar_dollar_return_var() {
        let input = "$$bills";
        let mut token_position = 0;
        let mut ast = SyntaxTree::default();
        let tokens = tokenize(input).unwrap();

        let _root = parse(&tokens, &mut token_position, input, &mut ast, u32::MIN).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![Node::Var {
                    data: "$$bills".to_string()
                },]
            }
        );
    }

    #[test]
    fn test_parse_bytes_returns_bytes_base85() {
        let input = "~~85'K|(_";
        let mut token_position = 0;
        let mut ast = SyntaxTree::default();
        let tokens = tokenize(input).unwrap();

        let _root = parse(&tokens, &mut token_position, input, &mut ast, u32::MIN).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![Node::Bytes {
                    data: "ABC".as_bytes().to_vec()
                },]
            }
        );
    }

    #[test]
    fn test_parse_bytes_returns_bytes_base64() {
        let input = "~~64'QUJD";
        let mut token_position = 0;
        let mut ast = SyntaxTree::default();
        let tokens = tokenize(input).unwrap();

        let _root = parse(&tokens, &mut token_position, input, &mut ast, u32::MIN).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![Node::Bytes {
                    data: "ABC".as_bytes().to_vec()
                },]
            }
        );
    }

    #[test]
    fn test_parse_binary_add_returns_add() {
        let input = "1+2";
        let mut token_position = 0;
        let mut ast = SyntaxTree::default();
        let tokens = tokenize(input).unwrap();

        let _root = parse(&tokens, &mut token_position, input, &mut ast, u32::MIN).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Int { data: 1 },
                    Node::Int { data: 2 },
                    Node::Add {
                        left: NodeId(0),
                        right: NodeId(1)
                    }
                ]
            }
        );
    }

    #[test]
    fn test_parse_binary_sub_returns_sub() {
        let input = "1-2";
        let mut token_position = 0;
        let mut ast = SyntaxTree::default();
        let tokens = tokenize(input).unwrap();

        let _root = parse(&tokens, &mut token_position, input, &mut ast, u32::MIN).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Int { data: 1 },
                    Node::Int { data: 2 },
                    Node::Sub {
                        left: NodeId(0),
                        right: NodeId(1)
                    }
                ]
            }
        );
    }

    #[test]
    fn test_parse_binary_add_right_returns_add() {
        let input = "1+2+3";
        let mut token_position = 0;
        let mut ast = SyntaxTree::default();
        let tokens = tokenize(input).unwrap();

        let _root = parse(&tokens, &mut token_position, input, &mut ast, u32::MIN).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Int { data: 1 },
                    Node::Int { data: 2 },
                    Node::Add {
                        left: NodeId(0),
                        right: NodeId(1)
                    },
                    Node::Int { data: 3 },
                    Node::Add {
                        left: NodeId(2),
                        right: NodeId(3)
                    }
                ]
            }
        );
    }

    #[test]
    fn test_parse_binary_add_right_returns_mul() {
        let input = "1+2*3";
        let mut token_position = 0;
        let mut ast = SyntaxTree::default();
        let tokens = tokenize(input).unwrap();

        let _root = parse(&tokens, &mut token_position, input, &mut ast, u32::MIN).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Int { data: 1 },
                    Node::Int { data: 2 },
                    Node::Int { data: 3 },
                    Node::Mul {
                        left: NodeId(1),
                        right: NodeId(2)
                    },
                    Node::Add {
                        left: NodeId(0),
                        right: NodeId(3)
                    }
                ]
            }
        );
    }

    #[test]
    fn test_parse_binary_add_right_returns_mul_left() {
        let input = "1*2+3";
        let mut token_position = 0;
        let mut ast = SyntaxTree::default();
        let tokens = tokenize(input).unwrap();

        let _root = parse(&tokens, &mut token_position, input, &mut ast, u32::MIN).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Int { data: 1 },
                    Node::Int { data: 2 },
                    Node::Mul {
                        left: NodeId(0),
                        right: NodeId(1)
                    },
                    Node::Int { data: 3 },
                    Node::Add {
                        left: NodeId(2),
                        right: NodeId(3)
                    }
                ]
            }
        );
    }

    #[test]
    fn test_exp_binds_tighter_than_mul_right() {
        let input = "5*2^3";
        let mut token_position = 0;
        let mut ast = SyntaxTree::default();
        let tokens = tokenize(input).unwrap();

        let _root = parse(&tokens, &mut token_position, input, &mut ast, u32::MIN).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Int { data: 5 },
                    Node::Int { data: 2 },
                    Node::Int { data: 3 },
                    Node::Exp { left: NodeId(1), right: NodeId(2) },
                    Node::Mul {
                        left: NodeId(0),
                        right: NodeId(3)
                    }
                ]
            }
        );
    }

    #[test]
    fn test_list_access_binds_tighter_than_append() {
        let input = "a +< ls@0";
        let mut token_position = 0;
        let mut ast = SyntaxTree::default();
        let tokens = tokenize(input).unwrap();

        let _root = parse(&tokens, &mut token_position, input, &mut ast, u32::MIN).unwrap();
        ast.dump_dot(_root, "/tmp/dump");

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Var{ data: "a".to_string() },
                    Node::Var{ data: "ls".to_string() },
                    Node::Int { data: 0 },
                    Node::Access { left: NodeId(1), right: NodeId(2) },
                    Node::ListAppend{ left: NodeId(0), right: NodeId(3) },
                ]
            }
        );
    }

    #[test]
    fn test_parse_binary_str_concat() {
        let input = "abc ++ def";
        let mut token_position = 0;
        let mut ast = SyntaxTree::default();
        let tokens = tokenize(input).unwrap();
        dbg!(&tokens);

        let _root = parse(&tokens, &mut token_position, input, &mut ast, u32::MIN).unwrap();
        ast.dump_dot(_root, "/tmp/dump");

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Var{ data: "abc".to_string() },
                    Node::Var{ data: "def".to_string() },
                    Node::StrConcat { left: NodeId(0), right: NodeId(1) },
                ]
            }
        );
    }
}
