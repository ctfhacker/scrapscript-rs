#![feature(concat_idents)]
#![feature(let_chains)]

mod colors;
use colors::Colorized;

use core::ops::{Index, IndexMut};
use std::collections::{HashSet, HashMap, VecDeque};

/// Error margin when comparing floats
const FLOAT_ERROR_MARGIN: f64 = 0.000_000_000_001;

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
    OpAnd,

    /// Operator ||
    OpOr,

    /// Operator ++
    OpStrConcat,

    /// Operator >+ (List concat)
    OpListCons,

    /// Operator +< (List append)
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

impl TryFrom<&str> for TokenId {
    type Error = TokenError;

    fn try_from(input: &str) -> Result<TokenId, Self::Error> {
        match input {
            "+" => Ok(TokenId::OpAdd),
            "-" => Ok(TokenId::OpSub),
            "*" => Ok(TokenId::OpMul),
            "/" => Ok(TokenId::OpDiv),
            "^" => Ok(TokenId::OpExp),
            "%" => Ok(TokenId::OpMod),
            "==" => Ok(TokenId::OpEqual),
            "/=" => Ok(TokenId::OpNotEqual),
            "<" => Ok(TokenId::OpLess),
            ">" => Ok(TokenId::OpGreater),
            "<=" => Ok(TokenId::OpLessEqual),
            ">=" => Ok(TokenId::OpGreaterEqual),
            "&&" => Ok(TokenId::OpAnd),
            "||" => Ok(TokenId::OpOr),
            "++" => Ok(TokenId::OpStrConcat),
            ">+" => Ok(TokenId::OpListCons),
            "+<" => Ok(TokenId::OpListAppend),
            x => Err(TokenError::UnknownCharacter(x.chars().nth(0).unwrap()))
        }
    }
}

impl TokenId {
    /// Provides operator precedence. The two values given are:
    ///
    /// * The value passed into the recursive parse function
    /// * The value checked to determine which operator has precedence
    ///
    /// The two values allow for having different precedence values for comparing the 
    /// same operation back-to-back. 
    ///
    /// If the value is >=, we have a right leaning tree.
    /// If the value is <,  we have a left leaning tree.
    ///
    /// Example:
    ///
    /// Where - For two WHERE operations, back to back, we want the first to have more precendence
    ///
    /// ```ignore
    /// a . b . c
    /// ````
    /// ```ignore
    ///      .
    ///     / \
    ///    .   c
    ///   / \
    ///  a   b
    /// ```
    ///
    /// Function - For two FUNCTION operations, back to back, we want the second to have more precedence
    ///
    /// ```ignore
    /// a -> b -> c + d
    /// ```
    /// ```ignore
    ///       ->
    ///      /  \
    ///     a   ->
    ///        /  \
    ///       b    +
    ///           / \
    ///          c   d
    /// ```
    #[must_use]
    pub fn get_op_precedence(&self) -> Option<(u32, u32)> {
        #[allow(clippy::enum_glob_use)]
        use TokenId::*;

        match self {
            OpAccess => Some((1001, 1001)),
            OpComposeReverse | OpCompose => Some((140, 140)),
            OpExp => Some((130, 130)),
            OpMul | OpDiv | OpFloorDiv | OpMod => Some((120, 120)),
            OpSub | OpAdd => Some((110, 110)),
            OpListCons | OpStrConcat => Some((100, 100)),
            OpListAppend => Some((100, 99)),
            OpEqual | OpNotEqual | OpLess | OpGreater | OpLessEqual | OpGreaterEqual => Some((90, 90)),
            OpAnd => Some((80, 80)),
            OpOr => Some((70, 70)),
            OpPipe => Some((60, 59)),
            OpReversePipe => Some((60, 60)),
            OpFunction => Some((50, 51)),
            OpMatchCase => Some((42, 42)),
            OpHasType => Some((41, 41)),
            OpAssign => Some((40, 40)),
            OpAssert => Some((32, 31)),
            OpWhere => Some((31, 30)),
            OpRightEval => Some((29, 29)),
            Comma => Some((10, 10)),
            OpSpread => Some((0, 0)),
            x => {
                println!("Unknown operator prescedence: {x:?}");
                None
            }
        }
    }
}

/// Abstract syntax tree nodes
#[derive(Debug, Clone, PartialEq)]
pub enum Node {
    /// A hole node
    Hole,

    /// A true node
    True,

    /// A false node
    False,

    /// An integer value (leaf)
    Int { data: i64 },

    /// A float value (leaf)
    Float { data: f64 },

    /// A string
    String { data: String },

    /// A variable
    Var { data: String },

    /// A symbol name
    Symbol { data: String },

    /// A vec of bytes
    Bytes { data: Vec<u8> },

    /// A vec of nodes
    List { data: Vec<NodeId> },

    /// A set of match cases
    MatchFunction { data: Vec<NodeId> },

    /// A record of nodes
    Record { keys: Vec<NodeId>, values: Vec<NodeId> },

    /// An assign operation
    Assign { left: NodeId, right: NodeId },

    /// A exponentiation operation
    Exp { left: NodeId, right: NodeId },

    /// A subtraction operation between two nodes
    Sub { left: NodeId, right: NodeId },

    /// An addition operation between two nodes
    Add { left: NodeId, right: NodeId },

    /// A multiplication operation between two nodes
    Mul { left: NodeId, right: NodeId },

    /// A divide operation between two nodes
    Div { left: NodeId, right: NodeId },

    /// A floor divide operation between two nodes
    FloorDiv { left: NodeId, right: NodeId },

    /// A modulo operation between two nodes
    Mod { left: NodeId, right: NodeId },

    /// A greater than operation between two nodes
    GreaterThan { left: NodeId, right: NodeId },

    /// A greater than or equal operation between two nodes
    GreaterEqual { left: NodeId, right: NodeId },

    /// A less than operation between two nodes
    LessThan { left: NodeId, right: NodeId },

    /// A less than or equal than operation between two nodes
    LessEqual { left: NodeId, right: NodeId },

    /// An equal operation between two nodes
    Equal { left: NodeId, right: NodeId },

    /// A not equal operation between two nodes
    NotEqual { left: NodeId, right: NodeId },

    /// An access operation between two nodes
    Access { left: NodeId, right: NodeId },

    /// An bitwise and operation between two nodes
    And { left: NodeId, right: NodeId },

    /// An bitwise or operation between two nodes
    Or { left: NodeId, right: NodeId },

    /// An apply operation between two nodes
    Apply { func: NodeId, arg: NodeId },

    /// A list append operation between two nodes
    ListAppend { left: NodeId, right: NodeId },

    /// A string concatination operation between two nodes
    StrConcat  { left: NodeId, right: NodeId },

    /// A list cons operation between two nodes
    ListCons  { left: NodeId, right: NodeId },

    /// A where operation
    Where { left: NodeId, right: NodeId },

    /// An assert operation
    Assert { left: NodeId, right: NodeId },

    /// A hastype operation
    HasType { left: NodeId, right: NodeId },

    /// A right eval operation
    RightEval { left: NodeId, right: NodeId },

    /// A function operation
    Function { arg: NodeId, body: NodeId },

    /// A match case
    MatchCase { left: NodeId, right: NodeId },

    /// A compose case
    Compose { left: NodeId, right: NodeId },

    /// A spread node
    Spread { name: Option<String> },

    /// A closure of an expression using a given environment
    Closure { env: HashMap<String, NodeId>, expr: NodeId }
}

impl From<f64> for Node {
    fn from(data: f64) -> Node { 
        Node::Float { data }
    }
}

impl From<i64> for Node {
    fn from(data: i64) -> Node { 
        Node::Int { data }
    }
}

impl Node {
    #[must_use]
    pub fn label(&self) -> String {
        match self {
            Node::True => "true".to_string(),
            Node::False => "false".to_string(),
            Node::Int { data } => format!("{data:#x}"),
            Node::Float { data } => format!("{data:.4}"),
            Node::List { .. } => {
                "LIST".to_string()
            }
            Node::Record { .. } => {
                "RECORD".to_string()
            }
            Node::Var { data }  | Node::Symbol { data } | Node::String { data } => data.to_string(),
            Node::Bytes { data } => {
                format!("{:?}", data.iter().map(|x| *x as char).collect::<Vec<_>>())
            }
            Node::Sub { .. } => "-".to_string(),
            Node::Add { .. } => "+".to_string(),
            Node::Mul { .. } => "*".to_string(),
            Node::Exp { .. } => "^".to_string(),
            Node::Div { .. } => "/".to_string(),
            Node::FloorDiv { .. } => "//".to_string(),
            Node::Mod { .. } => "%".to_string(),
            Node::GreaterThan { .. } => ">".to_string(),
            Node::GreaterEqual { .. } => ">=".to_string(),
            Node::LessThan { .. } => "<".to_string(),
            Node::LessEqual { .. } => "<=".to_string(),
            Node::Equal { .. } => "==".to_string(),
            Node::NotEqual { .. } => "/=".to_string(),
            Node::And { .. } => "&& (AND)".to_string(),
            Node::Or { .. } => "|| (OR)".to_string(),
            Node::Access { .. } => "@ (ACCESS) ".to_string(),
            Node::Apply { .. } => "APPLY".to_string(),
            Node::ListAppend { .. } => "+< (LIST_APPEND)".to_string(),
            Node::StrConcat { .. } => "++ (STR_CONCAT)".to_string(),
            Node::ListCons { .. } => ">+ (LIST_CONS)".to_string(),
            Node::Assign { .. } => "= (ASSIGN)".to_string(),
            Node::Function { .. } => "FUNCTION".to_string(),
            Node::Where { .. } => ". (WHERE)".to_string(),
            Node::Assert { .. } => "? (ASSERT)".to_string(),
            Node::HasType { .. } => ": (HAS_TYPE)".to_string(),
            Node::RightEval { .. } => "! (RIGHT_EVAL)".to_string(),
            Node::MatchCase { .. } => "case".to_string(),
            Node::MatchFunction{ .. } => "MATCH_FUNCTION".to_string(),
            Node::Compose { .. } => "<< (COMPOSE)".to_string(),
            Node::Hole => "()".to_string(),
            Node::Closure { .. } => "CLOSURE".to_string(),
            Node::Spread { name } => {
                let mut label = "...".to_string();
                if let Some(name) = name {
                    label.push_str(name);
                }
                label
            }
        }
    }
}


#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Type {
    True,
    False,
    Int,
    Float,
    String,
    Bytes,
    Var,
    List,
    Add,
    Sub,
    Mul,
    Div,
    FloorDiv ,
    Exp ,
    Mod ,
    Equal ,
    NotEqual,
    Assign,
    Function,
    StrConcat,
    ListCons,
    ListAppend,
    Where,
    Assert,
    Hole,
    Apply,
    Record ,
    Access,
    MatchFunction,
    Compose,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Token {
    id: TokenId,
    pos: u32,
}

impl Token {
    #[must_use]
    pub fn new(id: TokenId, pos: u32) -> Self {
        Self { id, pos }
    }
}


/// Errors that can occur during execution of scrapscript
#[derive(Debug)]
pub enum Error {
    /// An error occured during tokenization
    Tokenize((TokenError, usize)),

    /// An error occured during parsing
    Parse((ParseError, usize)),

    /// An error occured during evaluation
    Eval((EvalError, usize))
}

impl Error {
    /// The help message to display for this error
    #[must_use]
    pub fn help(&self) -> Option<String> {
        match self {
            Error::Eval((EvalError::VariableNotFoundInScope(var), _)) => {
                Some(format!("Was variable `{var}` defined before this use?"))
            }
            _ => None
        }
    }

    /// The note message to display for this error
    #[must_use]
    pub fn note(&self) -> Option<String> {
        match self {
            Error::Eval((EvalError::VariableNotFoundInScope(_), _)) => {
                Some("This is testing a specific note.".to_string())
            }
            _ => None
        }
    }
}

impl From<(TokenError, usize)> for Error {
    fn from(err: (TokenError, usize)) ->  Error {
        Error::Tokenize(err)
    }
}

impl From<(ParseError, usize)> for Error {
    fn from(err: (ParseError, usize)) ->  Error {
        Error::Parse(err)
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
    UnknownCharacter(char),
}

#[derive(Debug)]
pub enum ParseError {
    InternalParseError,
    UnknownToken(TokenId),
    UnknownOperatorToken(TokenId),
    NoTokensGiven,
    UnexpectedEndOfFile,

    Int(std::num::ParseIntError),
    Float(std::num::ParseFloatError),
    Base85(base85::Error),
    Base64(base64::DecodeError),

    /// Comma in list found before another valid token.
    /// Ex: [,] or [,,]
    ListCommaBeforeToken,
    RecordCommaBeforeAssignment,

    /// Left side of an assignment was not a variable
    NonVariableInAssignment,

    /// The expression in the record was not as assign operation
    NonAssignmentInRecord(TokenId),

    /// The expression in the record was not a function operation
    NonFunctionInMatch(TokenId),

    /// The name of a spread must be a name
    NonVariableSpreadNameFound,

    /// Parsed a non-leaf node as a first token
    NonLeafNodeFoundFirst,

    /// Spread must come at the end of a record/list
    SpreadMustComeAtEndOfCollection,

    /// Token other than Int or Float found during a subtraction
    NonIntFloatForSubtract,

    /// Different types found for the same list
    ListTypeMismatch(TokenId, TokenId),

    /// Invalid function type given to apply
    InvalidFunctionTypeInApply(Type),
}

macro_rules! impl_from_err {
    ($e:ty, $err:ident, $ty:ty) => {
        impl From<$ty> for $e {
            fn from (err: $ty) -> $e {
                <$e>::$err(err)
            }
        }
    }
}

impl_from_err!(ParseError, Int, std::num::ParseIntError);
impl_from_err!(ParseError, Float, std::num::ParseFloatError);
impl_from_err!(ParseError, Base85, base85::Error);
impl_from_err!(ParseError, Base64, base64::DecodeError);

/// A basic syntax tree of nodes
#[derive(Default, Debug, PartialEq)]
pub struct SyntaxTree {
    nodes: Vec<Node>,
    positions: Vec<u32>
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
        pub fn $func_name(&mut self, data: $ty, pos: u32) -> NodeId {
            let node = Node::$node { data };
            self.add_node(node, pos)
        }
    };
}

/// Implement a node that has `left` and `right` children
macro_rules! impl_left_right_node {
    ($func_name:ident, $node:ident) => {
        pub fn $func_name(&mut self, left: NodeId, right: NodeId, pos: u32) -> NodeId {
            let node = Node::$node { left, right };
            self.add_node(node, pos)
        }
    };
}

/// Returns `true` for a valid symbol character and `false` otherwise
/// 
/// Valid symbol characters
///
/// 'a'..='z' | 'A'..='Z' | '$' | '_' | '\''
#[must_use]
pub fn is_valid_symbol_byte(c: u8) -> bool {
    matches!(c, b'a'..=b'z' | b'A'..=b'Z' | b'$' | b'_' | b'\'')
}

/// Returns `true` for a valid name character and `false` otherwise
/// 
/// Valid symbol characters
///
/// 'a'..='z' | 'A'..='Z' | '0'..='9' | '$' | '_' | '\''
#[must_use]
pub fn is_valid_name_byte(c: u8) -> bool {
    matches!(c, b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'$' | b'_' | b'\'')
}

#[must_use]
pub fn is_not_double_quote(c: u8) -> bool {
    matches!(c, b'"')
}

/// Returns `true` for a valid int character and `false` otherwise
/// 
/// Valid symbol characters
///
/// '0'..='9'
#[must_use]
pub fn is_valid_int_byte(c: u8) -> bool {
    c.is_ascii_digit()
}

/// Returns `true` for a valid float character and `false` otherwise
/// 
/// Valid symbol characters
///
/// '0'..='9'
#[must_use]
pub fn is_valid_float_byte(c: u8) -> bool {
    matches!(c, b'0'..=b'9' | b'.')
}

/// Returns `true` for a valid base85 character and `false` otherwise
#[must_use]
pub fn is_valid_base85_byte(c: u8) -> bool {
    matches!(c, b'0'..=b'9' | b'A'..=b'Z' | b'a'..=b'z' | 
        b'!' | b'#' | b'$' | b'%' | b'&' | b'(' | b')' | b'*' | 
        b'+' | b'-' | b';' | b'<' | b'=' | b'>' | b'?' | b'@' | 
        b'^' | b'_' | b'`' | b'{' | b'|' | b'}' | b'~' | b'"')
}

/// Returns `true` for a valid base64 character and `false` otherwise
#[must_use]
pub fn is_valid_base64_byte(c: u8) -> bool {
    matches!(c, b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'=')
}

/// Returns `true` for a valid base32 character and `false` otherwise
#[must_use]
pub fn is_valid_base32_byte(c: u8) -> bool {
    matches!(c, b'A'..=b'Z' | b'2'..=b'7' | b'=')
}

/// Returns `true` for a valid base16 character and `false` otherwise
#[must_use]
pub fn is_valid_base16_byte(c: u8) -> bool {
    c.is_ascii_hexdigit()
}

impl SyntaxTree {
    impl_data_node!(int, Int, i64);
    impl_data_node!(float, Float, f64);
    impl_data_node!(name, Var, String);
    impl_data_node!(string, String, String);
    impl_data_node!(symbol, Symbol, String);
    impl_data_node!(bytes, Bytes, Vec<u8>);
    impl_data_node!(list, List, Vec<NodeId>);
    impl_left_right_node!(sub, Sub);
    impl_left_right_node!(add, Add);
    impl_left_right_node!(mul, Mul);
    impl_left_right_node!(exp, Exp);
    impl_left_right_node!(div, Div);
    impl_left_right_node!(floor_div, FloorDiv);
    impl_left_right_node!(modulo, Mod);
    impl_left_right_node!(greater_than, GreaterThan);
    impl_left_right_node!(greater_equal, GreaterEqual);
    impl_left_right_node!(less_than, LessThan);
    impl_left_right_node!(less_equal, LessEqual);
    impl_left_right_node!(equal, Equal);
    impl_left_right_node!(not_equal, NotEqual);
    impl_left_right_node!(and, And);
    impl_left_right_node!(or, Or);
    impl_left_right_node!(access, Access);
    impl_left_right_node!(list_append, ListAppend);
    impl_left_right_node!(str_concat, StrConcat);
    impl_left_right_node!(list_cons, ListCons);
    impl_left_right_node!(assign, Assign);
    impl_left_right_node!(where_op, Where);
    impl_left_right_node!(assert, Assert);
    impl_left_right_node!(has_type, HasType);
    impl_left_right_node!(right_eval, RightEval);
    impl_left_right_node!(match_case, MatchCase);
    impl_left_right_node!(compose, Compose);

    /// Add the given [`Node`] at `pos` into this [`SyntaxTree`]
    pub fn add_node(&mut self, node: Node, pos: u32) -> NodeId {
        let node_index = self.nodes.len();
        self.nodes.push(node);
        self.positions.push(pos);

        NodeId(node_index)
    }

    pub fn hole(&mut self, pos: u32) -> NodeId {
        let node = Node::Hole;
        self.add_node(node, pos)
    }

    pub fn spread(&mut self, name: Option<String>, pos: u32) -> NodeId {
        let node = Node::Spread { name };
        self.add_node(node, pos)
    }

    /// Insert a record node with the given `keys` and `values`
    pub fn record(&mut self, keys: Vec<NodeId>, values: Vec<NodeId>, pos: u32) -> NodeId {
        let node = Node::Record { keys, values };
        self.add_node(node, pos)
    }

    /// Insert an Apply node with the given `func` and `arg`
    pub fn apply(&mut self, func: NodeId, arg: NodeId, pos: u32) -> NodeId {
        let node = Node::Apply { func, arg };
        self.add_node(node, pos)
    }

    /// Insert a function node with the given `name` and `body`
    pub fn function(&mut self, name: NodeId, body: NodeId, pos: u32) -> NodeId {
        let node = Node::Function { arg: name, body};
        self.add_node(node, pos)
    }

    pub fn match_function(&mut self, data: Vec<NodeId>, pos: u32) -> NodeId {
        let node = Node::MatchFunction { data };
        self.add_node(node, pos)
    }

    /// Insert a closure node with the given `env` 
    pub fn closure(&mut self, env: HashMap<String, NodeId>, expr: NodeId, pos: u32) -> NodeId {
        let node = Node::Closure { env, expr };
        self.add_node(node, pos)
    }

    /// Insert a `true` node
    pub fn true_node(&mut self, pos: u32) -> NodeId {
        let node = Node::True;
        self.add_node(node, pos)
    }

    /// Insert a `true` node
    pub fn false_node(&mut self, pos: u32) -> NodeId {
        let node = Node::False;
        self.add_node(node, pos)
    }

    /// Dump a .dot of this syntax tree
    ///
    /// # Errors
    /// 
    /// * Failed to write the dot file
    ///
    /// # Panics
    /// 
    /// * TODO: Invalid match function
    #[allow(clippy::too_many_lines)]
    pub fn dump_dot(&self, root: NodeId, out_name: &str) -> Result<(), std::io::Error> {
        let mut queue = vec![root];
        let mut seen_nodes = HashSet::new();

        let mut dot = String::from("digraph {\n");
        dot.push_str("node [shape=record];\n");

        while let Some(node_id) = queue.pop() {
            // Ignore nodes we've already seen
            if !seen_nodes.insert(node_id) {
                continue;
            }

            let curr_node = &self.nodes[node_id];

            // List nodes are special
            if !matches!(curr_node, Node::List {.. } | Node::Record { .. } | Node::MatchFunction { .. }) {
                dot.push_str(&format!("node{node_id} [ label = {:?} ];\n", curr_node.label()));
            }

            match curr_node {
                Node::Sub { left, right }
                | Node::Add { left, right }
                | Node::Mul { left, right }
                | Node::Exp { left, right }
                | Node::Div { left, right }
                | Node::FloorDiv { left, right }
                | Node::Mod { left, right }
                | Node::ListAppend { left, right }
                | Node::StrConcat { left, right }
                | Node::ListCons { left, right }
                | Node::GreaterThan { left, right }
                | Node::GreaterEqual{ left, right }
                | Node::LessEqual { left, right }
                | Node::LessThan { left, right }
                | Node::Equal { left, right }
                | Node::NotEqual { left, right }
                | Node::And { left, right }
                | Node::Or { left, right }
                | Node::Assign { left, right }
                | Node::Where { left, right }
                | Node::Assert { left, right }
                | Node::HasType { left, right }
                | Node::RightEval { left, right }
                | Node::MatchCase{ left, right }
                | Node::Compose { left, right }
                | Node::Access { left, right } => {
                    queue.push(*left);
                    queue.push(*right);

                    dot.push_str(&format!("node{node_id} -> node{left}  [ label=\"left\"; ];\n"));
                    dot.push_str(&format!("node{node_id} -> node{right} [ label=\"right\"; ];\n"));
                }
                Node::Function { arg: name, body} => {
                    queue.push(*name);
                    queue.push(*body);

                    dot.push_str(&format!("node{node_id} -> node{name}  [ label=\"name\"; ];\n"));
                    dot.push_str(&format!("node{node_id} -> node{body} [ label=\"body\"; ];\n"));
                }
                Node::Apply { func, arg } => {
                    queue.push(*func);
                    queue.push(*arg);

                    dot.push_str(&format!("node{node_id} -> node{func}  [ label=\"func\"; ];\n"));
                    dot.push_str(&format!("node{node_id} -> node{arg} [ label=\"arg\"; ];\n"));
                }
                Node::List { data} => {
                    let label = data
                        .iter()
                        .enumerate().map(|(i, _)| format!("<f{i}> .{i}")).collect::<Vec<_>>().join(" | ");

                    dot.push_str(&format!("node{node_id} [ label = \"{label}\" ];\n"));

                    for (i, child_node_id) in data.iter().enumerate() {
                        queue.push(*child_node_id);
                        dot.push_str(&format!("node{node_id}:f{i} -> node{child_node_id}\n"));
                    }
                }
                Node::Record { keys, values } => {
                    let label = keys
                        .iter()
                        .enumerate()
                        .map(|(i, value)| format!("<f{i}> .{}", self.nodes[*value].label())).collect::<Vec<_>>().join("|");

                    dot.push_str(&format!("node{node_id} [ label = \"{label}\" ];\n"));

                    for (i, child_node_id) in values.iter().enumerate() {
                        queue.push(*child_node_id);
                        dot.push_str(&format!("node{node_id}:f{i} -> node{child_node_id};\n"));
                    }
                }
                Node::MatchFunction { data} => {
                    let label = data
                        .iter()
                        .enumerate().map(|(i, value)| {
                            let Node::MatchCase { left, .. } = self.nodes[*value] else {
                                panic!("ERROR: Non-matchcase in match function");
                            };
                            let case = &self.nodes[left];

                            format!("<f{i}> case {}", case.label())
                        }).collect::<Vec<_>>().join(" | ");

                    let label = format!("MATCH | {label}");

                    dot.push_str(&format!("node{node_id} [ label = \"{label}\" ];\n"));

                    for (i, child_node_id) in data.iter().enumerate() {
                        let Node::MatchCase { right, .. } = self.nodes[*child_node_id] else {
                            panic!("Non-matchcase in match function");
                        };

                        queue.push(right);
                        dot.push_str(&format!("node{node_id}:f{i} -> node{right}\n"));
                    }
                }
                Node::Closure { env, expr } => {
                    use std::fmt::Write;

                    dbg!(&env);
                    queue.push(*expr);

                    let label = env.iter()
                        .fold(String::new(), |mut acc, (key, node_id)| {
                            let _ = write!(acc, "{}={}", key, self.nodes[*node_id].label());
                            acc
                        });

                    dot.push_str(&format!("node [shape=record]; node{node_id} [ label = \"CLOSURE\nEnv:\n{label}\" ];\n"));
                    dot.push_str(&format!("node{node_id} -> node{expr}  [ label=\"expr\"; ];\n"));
                }
                Node::Int { .. } 
                | Node::Float { .. } 
                | Node::Var { .. } 
                | Node::String { .. } 
                | Node::Symbol { .. } 
                | Node::Bytes { .. } 
                | Node::Hole { .. }
                | Node::Spread  { .. }
                | Node::True { .. }
                | Node::False { .. }
                => {
                    // This is a leaf node.. nothing else to parse
                }
            }
        }

        dot.push('}');

        std::fs::write(out_name, dot)
    }

    /// Print the input with each node marked by a `v`
    #[allow(clippy::missing_panics_doc)]
    #[allow(clippy::cast_possible_truncation)]
    pub fn print_nodes(&self, input: &str) {
        let mut debug = vec![b' '; input.len()];
        input.chars()
             .enumerate()
             .filter(|(_index, byte)| *byte == '\n')
             .for_each(|(index, _byte)| debug[index] = b'\n');

        self.positions
            .iter()
            .filter(|i| **i < input.len() as u32)
            .for_each(|i| debug[*i as usize] = b'v');

        for (dbg, data) in debug.split(|x| *x == b'\n').zip(input.split(|x| x == '\n')) {
            println!("{}", std::str::from_utf8(dbg).unwrap());
            println!("{data}");
        }

    }

    #[must_use]
    pub fn get_type(&self, node: &NodeId) -> Type  {
        match &self.nodes[*node] {
            Node::Int { .. } => Type::Int,
            Node::Float { .. } => Type::Float,
            Node::String { .. } => Type::String,
            Node::List { .. } => Type::List,
            Node::Add { .. } => Type::Add,
            Node::Sub { .. } => Type::Sub,
            Node::Mul { .. } => Type::Mul,
            Node::Div { .. } => Type::Div,
            Node::FloorDiv { .. } => Type::FloorDiv ,
            Node::Exp { .. } => Type::Exp ,
            Node::Mod { .. } => Type::Mod ,
            Node::Equal { .. } => Type::Equal ,
            Node::NotEqual { .. } => Type::NotEqual,
            Node::Var { .. } => Type::Var,
            Node::ListCons { .. } => Type::ListCons,
            Node::ListAppend { .. } => Type::ListAppend,
            Node::StrConcat { .. } => Type::StrConcat,
            Node::Function { .. } => Type::Function,
            Node::Assign { .. } => Type::Assign,
            Node::Where  { .. } => Type::Where,
            Node::Assert { .. } => Type::Assert,
            Node::True { .. } => Type::True,
            Node::False { .. } => Type::False,
            Node::Hole { .. } => Type::Hole,
            Node::Apply { .. } => Type::Apply,
            Node::Record { .. } => Type::Record,
            Node::Access{ .. } => Type::Access,
            Node::MatchFunction { .. } => Type::MatchFunction,
            Node::Compose { .. } => Type::Compose,
            Node::Bytes { .. } => Type::Bytes,
            x => unimplemented!("Unknown type for node: {x:?}"),
            
        }
        
    }

    /// Get a reference to a node in the [`SyntaxTree`]
    #[must_use]
    pub fn get(&self, node: NodeId) -> &Node {
        &self.nodes[node]
    }

    #[must_use]
    #[allow(clippy::missing_panics_doc)]
    pub fn finish(&self, node: NodeId) -> Option<EvalResult> {
        // match self.nodes[node] {
        Some(match self.nodes.get(node.0)?.clone() {
            Node::Int { data } => EvalResult::Int(data),
            Node::Float { data } => EvalResult::Float(data.to_le_bytes()),
            Node::Assign { left, right } => {
                dbg!(&self.nodes[left]);
                dbg!(&self.nodes[right]);
                return None;
            }
            Node::String { data } => EvalResult::String(data),
            Node::True => EvalResult::True,
            Node::False => EvalResult::False,
            Node::List { data } => {
                let mut results = Vec::new();

                for node_id in data {
                    results.push(self.finish(node_id)?);
                }

                EvalResult::List(results)
            }
            Node::Record { keys, values } => {
                EvalResult::Record { 
                    keys: keys.iter().map(|k| self.finish(*k).unwrap()).collect(),
                    values: values.iter().map(|k| self.finish(*k).unwrap()).collect(),
                }
            }
            Node::StrConcat{ left, right } => {
                match (self.finish(left)?, self.finish(right)?) {
                    (EvalResult::String(mut left), EvalResult::String(right)) => {
                        left.push_str(&right);
                        EvalResult::String(left)
                    }
                    _ => return None
                }
            }
            Node::Bytes { data } => {
                EvalResult::Bytes(data)
            }
            x => todo!("{x:?}"),
        })
    }
}

/// The results of an evaluation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvalResult {
    True,
    False,
    Int(i64),
    Float([u8; 8]),
    String(String),
    List(Vec<EvalResult>),
    Bytes(Vec<u8>),
    Record { keys: Vec<EvalResult>, values: Vec<EvalResult> },
}

impl From<f64> for EvalResult {
    fn from(data: f64) -> EvalResult { 
        EvalResult::Float(data.to_le_bytes())
    }
}

impl From<i64> for EvalResult {
    fn from(data: i64) -> EvalResult { 
        EvalResult::Int(data)
    }
}

impl From<&str> for EvalResult {
    fn from(data: &str) -> EvalResult { 
        EvalResult::String(data.to_string())
    }
}

impl From<String> for EvalResult {
    fn from(data: String) -> EvalResult { 
        EvalResult::String(data)
    }
}

impl From<Vec<EvalResult>> for EvalResult {
    fn from(data: Vec<EvalResult>) -> EvalResult { 
        EvalResult::List(data)
    }
}

impl TryFrom<EvalResult> for f64 {
    type Error = &'static str;

    fn try_from(res: EvalResult) -> Result<f64, Self::Error> {
        match res {
            EvalResult::Float(bytes) => Ok(f64::from_le_bytes(bytes)),
            _ => Err("Not a valid f64")
        }
    }
    
}


#[derive(Debug, PartialEq, Clone)]
pub enum EvalError {
    InternalError,
    InvalidAssignmentVariable,
    InvalidNodeTypes(Type),
    ExponentPowerTooLarge(i64),
    NonStringFoundInStringConcat(Node),
    ListTypeMismatch(Type),

    /// The name of a function was not a `Var`
    InvalidFunctionName,

    /// Attempted to use a variable not found in the scope
    VariableNotFoundInScope(String),

    /// Evaluated a #false type in a condition
    FalseConditionFound,

    /// Function argument was not a variable
    NonVariableAsFunctionName,

    /// Function called with no available arguments
    NoArgumentsGivenToFunction,

    /// The requested access object was not found in the environment
    AccessObjectNotFound,

    /// The given key is not found in the record
    KeyNotFoundInRecord(String),

    /// The given type for list access was invalid
    InvalidListAccessType(Type),

    /// The given type requested for access was invalid
    InvalidAccessType(Type),

    /// The given index is out of bounds of the requested list
    ListIndexOutOfBounds { length: usize, wanted_index: usize },

    /// Requested match case not found in match function
    MatchCaseNotFound(Node, Vec<Node>)
}

#[derive(Debug, Default, Clone)]
pub struct Context {
    /// Variable assignments
    pub env: HashMap<String, NodeId>,

    /// Arguments passed to functions
    pub args: Vec<NodeId>

}

/// The evaluation state for an operation in the iterative case
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum EvalState {
    /// Needs its arguments to be evaluated
    NeedArguments,

    /// Can begin evaluation
    Ready,

    /// Needs its arguments to be evaluated with a new context
    NeedArgumentsNewContext,

    /// Needs its arguments to be evaluated with context currently on the context stack
    NeedArgumentsRestoreContext,
}

/// Evaluate the given node at `root` using the given context `ctx`
///
/// # Errors
/// 
/// * See [`EvalError`]
///
/// # Panics
/// 
/// * An error position cannot fit in `usize`
#[allow(clippy::too_many_lines)]
pub fn eval(
        ctx: &mut Context, 
        ast: &mut SyntaxTree, 
        start: NodeId) -> Result<EvalResult, (EvalError, u32)> {
    let mut work = VecDeque::new();
    work.push_front((start, EvalState::NeedArguments));

    // let mut stacks = VecDeque::new();
    let mut arguments = VecDeque::new();

    let mut contexts = VecDeque::new();

    'work: while let Some((curr_node_id, mut eval_state)) = work.pop_front() {
        let node_pos = ast.positions[curr_node_id.0];
        let curr_node = &ast.nodes[curr_node_id];

        eprintln!("--- TOP ---");
        print_ast(ast);
        eprintln!("Node ({curr_node_id}) {curr_node:?}");
        eprintln!("Contexts: {contexts:?}");
        eprintln!("State: {eval_state:?}");
        eprintln!("Env {:?}", ctx.env);

        match eval_state {
            EvalState::NeedArgumentsNewContext => {
                // Use a new context for this node
                let mut old_scope = Context::default();
                core::mem::swap(ctx, &mut old_scope);

                // Push the old contexts onto the stack of contexts
                contexts.push_front(old_scope);
                eval_state = EvalState::NeedArguments;
            }
            EvalState::NeedArgumentsRestoreContext => {
                // Restore an old context for this node
                let mut old_context = contexts.pop_front().unwrap();

                // Add the old context to the main context
                old_context.env.extend(ctx.env.drain());

                // Restore the original context
                *ctx = old_context;
                
                // Use the normal first state for this node
                eval_state = EvalState::NeedArguments;
            }
            /*
            EvalState::NeedArguments => {
                if !contexts.is_empty() {
                    // Restore an old context for this node
                    let mut old_context = contexts.pop_front().unwrap();

                    // Add the old context to the main context
                    // old_context.env.extend(ctx.env.drain());

                    // Restore the original context
                    *ctx = old_context;
                }
            }
            */
            _ => {}
        }

        eprintln!("Work {work:?}");
        eprintln!("Env {:?}", ctx.env);
        eprintln!("Arguments {arguments:?}");
        eprintln!("Contexts {contexts:?}");
       
        match curr_node {
            Node::Add { left, right }
            | Node::Sub { left, right }
            | Node::Mul { left, right }
            | Node::Div { left, right }
            | Node::FloorDiv { left, right }
            | Node::Exp { left, right }
            | Node::Mod { left, right }
            | Node::Equal { left, right }
            | Node::NotEqual {left, right } => {
                if eval_state == EvalState::NeedArguments {
                    work.push_front((curr_node_id, EvalState::Ready));
                    work.push_front((*left, EvalState::NeedArguments));
                    work.push_front((*right, EvalState::NeedArguments));
                    continue;
                }

                dbg!(&arguments);

                let left_data = arguments.pop_back().unwrap();
                let right_data = arguments.pop_back().unwrap();

                let left_data = ast.get(left_data);
                let right_data = ast.get(right_data);

                let node_type = ast.get_type(&curr_node_id);

                dbg!(left_data, right_data, node_type);

                match (left_data, right_data) {
                    (Node::Int { data: left_int}, Node::Int { data: right_int}) => {
                        // Adding two Ints
                        let data= match node_type {
                            Type::Add =>  left_int + right_int ,
                            Type::Sub =>  left_int - right_int ,
                            Type::Mul =>  left_int * right_int ,
                            Type::Mod =>  left_int % right_int ,
                            Type::Equal { .. } | Type::NotEqual { .. }=> {
                                let mut result = left_int == right_int;

                                // Invert the result for not equal
                                if matches!(node_type, Type::NotEqual { .. }) {
                                    result = !result;
                                }

                                if result {
                                    arguments.push_back(ast.true_node(node_pos));
                                } else {
                                    arguments.push_back(ast.false_node(node_pos));
                                }

                                continue;
                            }
                            Type::Div { .. } | Type::FloorDiv { .. } =>  left_int / right_int,
                            Type::Exp { .. } =>  {
                                let Ok(right) = (*right_int).try_into() else {
                                    return Err((EvalError::ExponentPowerTooLarge(*right_int), ast.positions[right.0]));
                                };

                                dbg!(left_int);
                                dbg!(right);
                                left_int.pow(right)
                            }
                            _ => unreachable!("{:?}", ast.nodes[curr_node_id])
                        };

                        dbg!(data);

                        arguments.push_back(ast.int(data, node_pos));
                    }
                    // Adding two Floats or a Float with a 0
                    (left @ (Node::Float { .. } | Node::Int { data: 0}), Node::Float { data: right }) => {
                        // A negative float is represented by a `0 - floatnum`. We need 
                        // to convert this zero to a floating point zero
                        let left = match left {
                            Node::Float { data} => *data,
                            Node::Int { data: 0 } => 0.0,
                            _ => unreachable!()
                        };

                        let right  = *right;

                        let data = match node_type {
                            Type::Add { .. } =>  left + right,
                            Type::Sub { .. } =>  left - right,
                            Type::Mul { .. } =>  left * right,
                            Type::Div { .. } =>  left / right,
                            Type::Mod { .. } =>  left % right,
                            Type::Exp { .. } =>  left.powf(right),
                            Type::FloorDiv { .. } =>  (left / right).floor(),
                            Type::NotEqual { .. } | Type::Equal { .. } => {
                                // 
                                let l_abs = left.abs();
                                let r_abs = right.abs();
                                let diff = (left - right).abs();

                                let mut result = if (left - right).abs() < FLOAT_ERROR_MARGIN {
                                    true
                                } else if left == 0.0 || right == 0.0 || (l_abs + r_abs < f64::MIN_POSITIVE) {
                                    diff < f64::EPSILON * f64::MIN_POSITIVE
                                } else {
                                    diff / (l_abs + r_abs).min(f64::MAX) < f64::EPSILON
                                };

                                // Invert the result for not equal
                                if matches!(node_type, Type::NotEqual { .. }) {
                                    result = !result;
                                }

                                if result {
                                    arguments.push_back(ast.true_node(node_pos));
                                }  else {
                                    arguments.push_back(ast.false_node(node_pos));
                                }

                                continue;
                            }
                            _ => unreachable!("{:?}", ast.nodes[curr_node_id])
                        };


                        arguments.push_back(ast.float(data, node_pos));
                    }

                    (Node::Var { data: var }, Node::Int { data: number}) | 
                    (Node::Int { data: number}, Node::Var { data: var }) => {
                        dbg!(var);
                        dbg!(&ctx.env);

                        // Check if the requested variable is in scope
                        let Some(var_node_id) = ctx.env.get(var) else {
                            let node_pos = if matches!(left_data, Node::Var { .. }) {
                                ast.positions[left.0]
                            } else {
                                ast.positions[right.0]
                            };

                            return Err((EvalError::VariableNotFoundInScope(var.to_string()), node_pos));
                        };

                        // Check that the variable is of the same type
                        let Node::Int { data: var_num } = ast.nodes[*var_node_id] else {
                            return Err((EvalError::InvalidNodeTypes(ast.get_type(var_node_id)), node_pos));
                        };

                        let left_int = var_num;
                        let right_int = *number;

                        // Calculate answer based on node type
                        let result= match node_type {
                            Type::Add =>  left_int + right_int ,
                            Type::Sub =>  left_int - right_int ,
                            Type::Mul =>  left_int * right_int ,
                            Type::Mod =>  left_int % right_int ,
                            Type::Equal { .. } | Type::NotEqual { .. }=> {
                                let mut result = left_int == right_int;

                                // Invert the result for not equal
                                if matches!(node_type, Type::NotEqual { .. }) {
                                    result = !result;
                                }

                                if result {
                                    return Ok(EvalResult::True);
                                } 

                                return Ok(EvalResult::False);
                            }
                            Type::Div { .. } | Type::FloorDiv { .. } =>  left_int / right_int,
                            Type::Exp { .. } =>  {
                                let Ok(right) = right_int.try_into() else {
                                    return Err((EvalError::ExponentPowerTooLarge(right_int), ast.positions[right.0]));
                                };

                                dbg!(left_int);
                                dbg!(right);
                                left_int.pow(right)
                            }
                            _ => unreachable!("{:?}", ast.nodes[curr_node_id])
                        };

                        dbg!(result);

                        arguments.push_back(ast.int(result, node_pos));
                    }
                    /*
                    (Node::Var { data: var_left }, Node::Var { data: var_right }) => {
                        // a + b

                        // Check if the requested variable is in scope
                        let Some(left_node_id) = ctx.env.get(var_left) else {
                            return Err((EvalError::VariableNotFoundInScope(var_left.to_string()), node_pos));
                        };
                        let left_node_id = *left_node_id;

                        // Check if the requested variable is in scope
                        let Some(right_node_id) = ctx.env.get(var_right) else {
                            return Err((EvalError::VariableNotFoundInScope(var_right.to_string()), node_pos));
                        };
                        let right_node_id = *right_node_id;

                        // Evaluate both sides of the addition
                        let left_data = eval(ctx, ast, left_node_id)?;
                        let right_data = eval(ctx, ast, right_node_id)?;

                        // Add 
                        let res_node = ast.add(left_data, right_data, node_pos);

                        let result = eval(ctx, ast, res_node)?;

                        arguments.push_back(result);
                    }
                    */
                    _ => {
                        if let Node::Add { left, right } = ast.nodes[curr_node_id] {
                            dbg!(&ctx);
                            dbg!(&ast.nodes[left]);
                            dbg!(&ast.nodes[right]);
                        }

                        return Err((EvalError::InvalidNodeTypes(ast.get_type(&curr_node_id)), node_pos));
                    }
                }
            }
            Node::Assign { left, right }  => {
                if eval_state == EvalState::NeedArguments {
                    work.push_front((curr_node_id, EvalState::Ready));
                    work.push_front((*right, EvalState::NeedArguments));
                    continue 'work;
                }

                let Node::Var { data: ref var} = ast.nodes[*left] else {
                    return Err((EvalError::InvalidAssignmentVariable, ast.positions[left.0]));
                };

                let env_key = var.to_string();
            
                if let Node::Function { body, .. } = ast.nodes[*right] && ast.nodes[body] == ast.nodes[*left] {
                    // Assigning a variable to a function with a body of that variable IS a closure

                    // Get the closure id ahead of time
                    let closure_id = ast.nodes.len();

                    // Create the closure env with its id in it
                    let mut closure_env = HashMap::new();
                    closure_env.insert(var.to_string(), NodeId(closure_id));

                    // Create the new closure
                    let closure_id = ast.closure(closure_env, *right, node_pos);

                    // Add the variable assignment to the environment
                    ctx.env.insert(env_key, closure_id);

                    // Return the closure as the result
                    arguments.push_back(closure_id);
                } else {
                    ctx.env.insert(env_key, *right);
                    // arguments.push_back(curr_node_id);
                } 
            }
            Node::Function { arg: name, body} => {
                if eval_state == EvalState::NeedArguments {
                    arguments.push_back(curr_node_id);
                    continue 'work;
                }

                let arg_name = if let Node::Var { data } = &ast.nodes[name.0] {
                    data.clone()
                } else {
                    return Err((EvalError::InvalidFunctionName, ast.positions[name.0]));
                };

                // Pop the last argument given
                let Some(argument) = arguments.pop_back() else {
                    return Err((EvalError::NoArgumentsGivenToFunction, node_pos));
                };

                let mut old_scope = Context::default();
                core::mem::swap(ctx, &mut old_scope);

                // Use this local scope for calling the body of this function
                contexts.push_front(old_scope);

                // Add this argument to the environment
                ctx.env.insert(arg_name.clone(), argument);

                // Special case where the body of a function is just a variable
                if let Node::Var { data } = &ast.nodes[body.0] {
                    let Some(var) = ctx.env.get(data) else {
                        return Err((EvalError::VariableNotFoundInScope(data.clone()), ast.positions[body.0]));
                    };

                    let var = *var;

                    // Add this argument to the environment
                    ctx.env.remove(&arg_name);

                    arguments.push_back(var);
                    continue;
                }

                work.push_front((*body, EvalState::NeedArgumentsRestoreContext));
            }
            Node::StrConcat { left, right } => {
                if eval_state == EvalState::NeedArguments {
                    work.push_front((curr_node_id, EvalState::Ready));
                    work.push_front((*left, EvalState::NeedArguments));
                    work.push_front((*right, EvalState::NeedArguments));
                    continue;
                }

                let left_data = arguments.pop_back().unwrap();
                let right_data = arguments.pop_back().unwrap();

                let left_data = ast.get(left_data);
                let right_data = ast.get(right_data);

                dbg!(&left_data);
                dbg!(&right_data);
            
                if let (Node::String { data: l_str }, Node::String { data: r_str}) = (left_data, right_data) {
                    let mut new_data = l_str.clone();
                    new_data.push_str(r_str);
                    arguments.push_back(ast.string(new_data, node_pos));
                } else {
                    return Err((EvalError::NonStringFoundInStringConcat(curr_node.clone()), node_pos));
                }
            }
            Node::List { data} => {
                if eval_state == EvalState::NeedArguments {
                    work.push_front((curr_node_id, EvalState::Ready));
                    for item in data {
                        work.push_front((*item, EvalState::NeedArguments));
                    }
                    continue;
                }

                // Pop the results back from the arguments list
                let results = (0..data.len()).map(|_| arguments.pop_back().unwrap()).collect();

                arguments.push_back(ast.list(results, node_pos));
            }
            Node::ListCons { left, right } => {
                if eval_state == EvalState::NeedArguments {
                    work.push_front((curr_node_id, EvalState::Ready));
                    work.push_front((*left, EvalState::NeedArguments));
                    work.push_front((*right, EvalState::NeedArguments));
                    continue;
                }

                let left_data_id = arguments.pop_back().unwrap();
                let right_data_id = arguments.pop_back().unwrap();

                let left_data = ast.get(left_data_id);
                let right_data = ast.get(right_data_id);

                dbg!(&left_data);
                dbg!(&right_data);

                match right_data {
                    Node::List { data } => {
                        let mut results = vec![left_data_id];

                        for d in data {
                            results.push(*d);
                        }

                        dbg!(&results);

                        arguments.push_back(ast.list(results, node_pos));
                    }
                    _ => {
                        return Err((EvalError::ListTypeMismatch(ast.get_type(&right_data_id)), ast.positions[right.0]));
                    }
                }
            }
            Node::ListAppend { left, right } => {
                if eval_state == EvalState::NeedArguments {
                    work.push_front((curr_node_id, EvalState::Ready));
                    work.push_front((*left, EvalState::NeedArguments));
                    work.push_front((*right, EvalState::NeedArguments));
                    continue;
                }

                let left_data_id = arguments.pop_back().unwrap();
                let right_data_id = arguments.pop_back().unwrap();

                let left_data = ast.get(left_data_id);
                let right_data = ast.get(right_data_id);

                dbg!(&left_data);
                dbg!(&right_data);

                match left_data {
                    Node::List { data } => {
                        let mut new_data = data.clone();
                        new_data.push(right_data_id);
                        arguments.push_back(ast.list(new_data, node_pos));
                    }
                    _ => {
                        return Err((EvalError::ListTypeMismatch(ast.get_type(&left_data_id)), ast.positions[left.0]));
                    }
                }
            }
            Node::Where { left, right } => {
                if eval_state == EvalState::NeedArguments {
                    // Right will be evaluated before the left
                    // 
                    // Right will be evalauted with a new context
                    // Left  will be evaluated using the context from the right's evaluation
                    // work.push_front((curr_node_id, EvalState::Ready));
                    work.push_front((*left, EvalState::NeedArguments));
                    work.push_front((*right, EvalState::NeedArguments));
                    continue;
                }
            }
            Node::Assert { left, right } => {
                if eval_state == EvalState::NeedArguments {
                    work.push_front((curr_node_id, EvalState::Ready));
                    work.push_front((*left, EvalState::NeedArguments));
                    work.push_front((*right, EvalState::NeedArguments));
                    continue;
                }

                let left_data_id = arguments.pop_back().unwrap();
                let right_data_id = arguments.pop_back().unwrap();

                let left_data = ast.get(left_data_id);
                let right_data = ast.get(right_data_id);

                dbg!(&left_data);
                dbg!(&right_data);

                match (left_data, right_data) {
                    (Node::True, _) =>  arguments.push_back(right_data_id),
                    (_, Node::True) =>  arguments.push_back(left_data_id),
                    (Node::False, _) => return Err((EvalError::FalseConditionFound, ast.positions[left.0])),
                    (_, Node::False) => return Err((EvalError::FalseConditionFound, ast.positions[right.0])),
                    _ => todo!()
                }
            }
            Node::Apply { func, arg } => {
                // Evaluate the argument, then the function (with a new context), then this apply again to restore the original context
                if eval_state == EvalState::NeedArguments {
                    work.push_front((curr_node_id, EvalState::Ready));
                    work.push_front((*func, EvalState::NeedArguments));
                    work.push_front((*arg, EvalState::NeedArguments));
                    continue;
                }

                let func = arguments.pop_back().unwrap();
                let arg = arguments.pop_back().unwrap();

                arguments.push_back(arg);
                work.push_front((func, EvalState::Ready));

                /*
                assert!(!contexts.is_empty());

                // Restore the original context
                let old_context = contexts.pop_front().unwrap();
                *ctx = old_context;
                */
            }
            Node::Record { keys, values } => {
                // Evaluate the argument, then the function (with a new context), then this apply again to restore the original context
                if eval_state == EvalState::NeedArguments {
                    work.push_front((curr_node_id, EvalState::Ready));

                    for (key, val) in keys.iter().zip(values.iter()) {
                        work.push_front((*key, EvalState::NeedArguments));
                        work.push_front((*val, EvalState::NeedArguments));
                    }

                    continue;
                }

                let length = keys.len();

                let mut new_keys = Vec::new();
                let mut new_values= Vec::new();

                for _ in 0..length {
                    new_keys.push(arguments.pop_back().unwrap());
                    new_values.push(arguments.pop_back().unwrap());
                }

                arguments.push_back(ast.record(new_keys, new_values, node_pos));
            }

            Node::Access { left, right } => {
                if eval_state == EvalState::NeedArguments {
                    work.push_front((curr_node_id, EvalState::Ready));
                    work.push_front((*left, EvalState::NeedArguments));
                    work.push_front((*right, EvalState::NeedArguments));
                    continue;
                }

                let left_data_id = arguments.pop_back().unwrap();
                let right_data_id = arguments.pop_back().unwrap();

                let left_data = ast.get(left_data_id);
                let right_data = ast.get(right_data_id);

                dbg!(&left_data);
                dbg!(&right_data);

                match left_data {
                    Node::Record { keys, values } => {
                        let Node::Var { data: wanted_key} = right_data else {
                            unreachable!();
                        };

                        for index in 0..keys.len() {
                            let Node::Var { data } = ast.get(keys[index]) else {
                                unreachable!();
                            };

                            dbg!(data, wanted_key);

                            // If this isn't the requested key, keep looking
                            if data != wanted_key {
                                continue;
                            }

                            arguments.push_back(values[index]);
                            continue 'work;
                        }

                        return Err((EvalError::KeyNotFoundInRecord(wanted_key.clone()), ast.positions[right.0]));
                    
                    }
                    Node::List { data } => {
                        let Node::Int { data: index} = right_data else {
                            return Err((EvalError::InvalidListAccessType(ast.get_type(right)), ast.positions[right.0]));
                        };

                        if usize::try_from(*index).unwrap() >= data.len() {
                            return Err((
                                EvalError::ListIndexOutOfBounds { 
                                    length: data.len(), 
                                    wanted_index: usize::try_from(*index).unwrap() 
                                }, ast.positions[right.0]));
                        }

                        let index: usize = (*index).try_into().unwrap();
                        arguments.push_back(data[index]);
                        continue
                    
                    }
                    _ => return Err((EvalError::InvalidAccessType(ast.get_type(&left_data_id)), ast.positions[left.0])),
                }
            }
            Node::MatchFunction { data} => {
                // A function needs to be used in [`Node::Apply`] operation. Until then, add itself to the argument list.
                if eval_state == EvalState::NeedArguments {
                    arguments.push_back(curr_node_id);
                    continue 'work;
                }

                let Some(argument) = arguments.pop_back() else {
                    return Err((EvalError::NoArgumentsGivenToFunction, node_pos));
                };

                // Get the requested match case
                let wanted_case = &ast.nodes[argument.0];
                let wanted_case_type = ast.get_type(&argument);

                // Check for the correct match case
                for assignment in data {
                    let Node::MatchCase { left, right } = ast.nodes[*assignment] else {
                        unreachable!();
                    };

                    // Otherwise, check if this exact case matches
                    let curr_check_case = &ast.nodes[left.0];

                    match curr_check_case {
                        Node::Record { keys: checked_keys, ..  } if wanted_case_type == Type::Record => {
                            let Node::Record { keys: wanted_keys, values: wanted_values } = wanted_case else {
                                unreachable!();
                            };

                            let mut found = true;
                            let mut results = HashSet::new();

                            // Iterate over the keys in the wanted case looking for a match in the possible match branches
                            'next_key: for (index, wanted_key_id) in wanted_keys.iter().enumerate() {
                                let wanted_key = &ast.nodes[wanted_key_id.0];

                                for checked_key_id in checked_keys {
                                    let checked_key = &ast.nodes[checked_key_id.0];
                                    if checked_key == wanted_key {
                                        let Node::Var { data} = checked_key else {
                                            unreachable!();
                                        };

                                        let new_value = wanted_values[index];
                                        results.insert((data.to_string(), new_value));
                                        continue 'next_key;
                                    }
                                }

                                found = false;
                                break 'next_key;
                            }

                            if found {
                                for (key, val) in results {
                                    ctx.env.insert(key, val);
                                }

                                work.push_front((right, EvalState::NeedArguments));
                                continue 'work;
                            }
                        
                            dbg!(curr_check_case);
                            todo!();
                        }

                        Node::Int { .. } => {
                            if wanted_case == curr_check_case {
                                arguments.push_back(right);
                                continue 'work;
                            }
                        }
                        // Variable match case matches everything
                        Node::Var { data } => {
                            ctx.env.insert(data.to_string(), argument);
                            work.push_front((right, EvalState::NeedArguments));
                            continue 'work;
                        }
                        x => {
                            unimplemented!("{x:?}");
                        }
                    }
                }


                // Match case NOT found.. populate the error result
                let mut all_cases = Vec::new();
                for assignment in data {
                    let Node::MatchCase { left, .. } = ast.nodes[*assignment] else {
                        unreachable!();
                    };

                    let curr_case = &ast.nodes[left.0];
                    all_cases.push(curr_case.clone());
                }

                return Err((EvalError::MatchCaseNotFound(wanted_case.clone(), all_cases), ast.positions[argument.0]));
            }
            Node::Compose { left, right } => {
                // Allow [`Node::Apply`] to call compose with Ready
                if eval_state == EvalState::NeedArguments {
                    arguments.push_back(curr_node_id);
                    continue 'work;
                }

                // Now that we are [`Ready`]
                work.push_front((*right, EvalState::Ready));
                work.push_front((*left, EvalState::Ready));
            }
            Node::Var { data } => {
                if let Some(var_node) = ctx.env.get(data) {
                    arguments.push_back(*var_node);
                } else {
                    // Otherwise, push the variable onto the arguments for something else to use
                    arguments.push_back(curr_node_id);
                }
            }
            Node::Int { .. }
            | Node::Float { .. }
            | Node::String { .. }
            | Node::Bytes { .. }
            | Node::Hole  { .. }
            | Node::True { .. }
            | Node::False { .. }
            => {
                // Base case.. Return the node as is. Nothing more to evaluate
                arguments.push_back(curr_node_id);
            }
            x => {
                unimplemented!("{x:?}");
            }
        }
    }

    // Error case for debugging
    if arguments.is_empty() {
        print_ast(ast);
        dbg!(&arguments);
        assert!(arguments.len() == 1);
    }

    let mut result = arguments.pop_back().unwrap();

    // If the result is a variable, attempt to use the variable's value
    if let Some(Node::Var { data }) = ast.nodes.get(result.0) {
        if let Some(res) = ctx.env.get(data) {
            result = *res;
            
        }
    }

    // Clear the context
    ctx.args.clear();
    ctx.env.clear();

    Ok(ast.finish(result).unwrap())
}

/// Convert the input program into a list of tokens
///
/// # Errors
///
/// * Token parsing error along with the position where the error occured
#[allow(clippy::too_many_lines)]
pub fn tokenize(input: &str) -> Result<Vec<Token>, (TokenError, usize)> {
    let mut result = Vec::new();
    let input = input.as_bytes();

    let mut index = 0;
    let mut in_token = false;

    let mut pos;

    
    'done: loop {
        macro_rules! check_eof {
            () => {
                if index >= input.len() {
                    result.push(Token {
                        id: TokenId::EndOfFile,
                        pos: input.len() as u32,
                    });
                    break 'done;
                }
            
            }
        
        }

        macro_rules! continue_while_space {
            () => {
                // Skip over all whitespace. Reset the current token when hitting whitespace.
                while matches!(input[index], b' ' | b'\n' | b'\r' | b'\t') {
                    index += 1;

                    check_eof!();

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
                    check_eof!();
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

        check_eof!();

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


                #[allow(clippy::cast_possible_truncation)]
                let pos = pos as u32;

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
                        result.push(Token::new(id, pos));

                        #[allow(clippy::cast_possible_truncation)]
                        result.push(Token {
                            id: TokenId::EndOfFile,
                            pos: input.len() as u32,
                        });
                        break 'done;
                    }
                }

                result.push(Token::new(id, pos));
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

                    check_eof!();
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
                set_token!(OpAnd, 2);
            }
            [b'|', b'|', ..] => {
                set_token!(OpOr, 2);
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
                while is_valid_symbol_byte(input[index]) {
                    index += 1;
                    is_empty = false;

                    check_eof!();
                }

                if is_empty {
                    if index >= input.len() {
                        return Err((TokenError::UnexpectedEndOfFile, pos));
                    }

                    return Err((TokenError::InvalidSymbol, pos));
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
            [b'[', ..] => {
                set_token!(LeftBracket, 1);
            }
            [b']', ..] => {
                set_token!(RightBracket, 1);
            }
            [b',', ..] => {
                set_token!(Comma, 1);
            }
            [b'{', ..] => {
                set_token!(LeftBrace, 1);
            }
            [b'}', ..] => {
                set_token!(RightBrace, 1);
            }
            [b'?', ..] => {
                set_token!(OpAssert, 1);
            }
            [b'-', ..] => {
                set_token!(OpSub, 1);

                // Reset the token for negative numbers -123 vs subtract of a - 123
                if input[index].is_ascii_digit() {
                    in_token = false;
                }
            }
            x => {
                return Err((TokenError::UnknownCharacter(x[0] as char), pos));
            }
        }
    }

    Ok(result)
}

/// Parse the given input string into a syntax tree
///
/// # Errors
///
/// * Error occurs during tokenization of input
/// * Error occurs during parsing of tokens
pub fn parse_str(input: &str) -> Result<(NodeId, SyntaxTree), Error> {
    let tokens = tokenize(input)?;
    // dbg!(&tokens);
    let (node, ast) = parse(&tokens, input)?;
    // dbg!(&ast);
    ast.print_nodes(input);

    Ok((node, ast))
}


/// Parse the given [`Token`]s into a [`SyntaxTree`]
///
/// # Errors
///
/// * Error occurs during parsing of tokens
pub fn parse(
    tokens: &[Token],
    input: &str,
) -> Result<(NodeId, SyntaxTree), (ParseError, usize)> {
    let mut ast = SyntaxTree::default();
    let mut token_position = 0;

    let root = _parse(tokens, &mut token_position, input, &mut ast, u32::MIN)?;

    Ok((root, ast))
}

/// Converts a list of [`Token`] into a [`SyntaxTree`] and returns the root node
///
/// # Errors
///
/// * Errors on parsing tokens and return the position of the error
#[allow(clippy::too_many_lines)]
#[allow(clippy::missing_panics_doc)]
pub fn _parse(
    tokens: &[Token],
    token_index: &mut usize,
    input: &str,
    ast: &mut SyntaxTree,
    current_precedence: u32,
) -> Result<NodeId, (ParseError, usize)> {
    if tokens.is_empty() {
        return Err((ParseError::NoTokensGiven, 0));
    }

    let parse_leaf = |ast: &mut SyntaxTree,
                      token_index: &mut usize|
     -> Result<NodeId, (ParseError, usize)> {
        let start_token_pos = *token_index;

        let start = tokens[start_token_pos];
        *token_index += 1;

        if *token_index >= tokens.len() {
            return Err((ParseError::UnexpectedEndOfFile, input.len()));
        }

        let input_index = start.pos as usize;
        let mut end_input_index = tokens[*token_index].pos as usize - 1;

        macro_rules! continue_while {
            ($check_func:ident) => {
                let input_bytes = input.as_bytes();

                // Skip over all whitespace. Reset the current token when hitting whitespace.
                while end_input_index > input_index && !$check_func(input_bytes[end_input_index]) {
                    end_input_index -= 1;
                }
            };
        }

        // ast.print_nodes(input);
        // print_ast(&ast);

        // Parse the next leaf token
        let leaf = match start.id {
            TokenId::OpSpread => {
                let mut next_token = tokens[*token_index];
                let mut spread_name = None;
                match next_token.id {
                    TokenId::RightBracket | TokenId::RightBrace => {
                        // Spread name will remain nameless
                    }
                    TokenId::Name => {
                        let input_index = next_token.pos as usize;
                        let mut end_input_index = tokens[*token_index + 1].pos as usize - 1;

                        let input_bytes = input.as_bytes();

                        // Skip over all whitespace. Reset the current token when hitting whitespace.
                        while end_input_index > input_index && !is_valid_name_byte(input_bytes[end_input_index]) {
                            end_input_index -= 1;
                        }

                        spread_name = Some(input[input_index..=end_input_index].to_string());

                        *token_index += 1;
                        next_token = tokens[*token_index];
                    }
                    _ => return Err((ParseError::SpreadMustComeAtEndOfCollection, start.pos.try_into().unwrap())),
                }

                if !matches!(next_token.id, TokenId::RightBracket | TokenId::RightBrace) {
                    return Err((ParseError::SpreadMustComeAtEndOfCollection, 1));
                }

                ast.spread(spread_name, start.pos)
            }
            TokenId::Int => {
                continue_while!(is_valid_int_byte);

                let value = input[input_index..=end_input_index]
                    .parse()
                    .map_err(|e| (ParseError::Int(e), input_index))?;

                // Add an Int node
                ast.int(value, input_index.try_into().unwrap())
            }
            TokenId::Float => {
                continue_while!(is_valid_float_byte);

                let value = input[input_index..=end_input_index]
                    .parse()
                    .map_err(|e| (ParseError::Float(e), input_index))?;

                // Add an Int node
                ast.float(value, input_index.try_into().unwrap())
            }
            TokenId::OpSub => {
                let zero = ast.int(0, u32::MAX);
                let right = _parse(tokens, token_index, input, ast, u32::MAX)?;
                ast.sub(zero, right, input_index.try_into().unwrap())
            }
            TokenId::Name => {
                continue_while!(is_valid_name_byte);
                let name = input[input_index..=end_input_index].to_string();
                ast.name(name, input_index.try_into().unwrap())
            }
            TokenId::String => {
                continue_while!(is_not_double_quote);
                let name = input[input_index + 1..end_input_index].to_string();
                ast.string(name, input_index.try_into().unwrap())
            }
            TokenId::Symbol => {
                continue_while!(is_valid_symbol_byte);

                // Start the symbol position at the # itself
                let pos = (input_index - 1).try_into().unwrap();

                match &input[input_index..=end_input_index] {
                    "true" => ast.true_node(pos),
                    "false" => ast.false_node(pos),
                    x => ast.symbol(x.to_string(), pos),
                }
            }
            TokenId::Base85 => {
                continue_while!(is_valid_base85_byte);
                let name = &input[input_index..=end_input_index];

                let bytes = base85::decode(name)
                    .map_err(|e| (ParseError::Base85(e), start.pos as usize))?;
                ast.bytes(bytes, input_index.try_into().unwrap())
            }
            TokenId::Base64 => {
                use base64::Engine;

                continue_while!(is_valid_base64_byte);
                let name = &input[input_index..=end_input_index];

                // Decode the base64 string
                let bytes = base64::engine::general_purpose::STANDARD
                    .decode(name)
                    .map_err(|e| (ParseError::Base64(e), start.pos as usize))?;

                // Create the decoded bytes string
                ast.bytes(bytes, input_index.try_into().unwrap())
            }
            TokenId::Base32 => {
                use data_encoding::BASE32;

                continue_while!(is_valid_base32_byte);
                let name = &input[input_index..=end_input_index];

                let mut bytes= vec![0; BASE32.decode_len(name.len()).unwrap()];

                let len = BASE32.decode_mut(name.as_bytes(), &mut bytes).unwrap();

                // Create the decoded bytes string
                ast.bytes(bytes[..len].to_vec(), input_index.try_into().unwrap())
            }
            TokenId::Base16 => {
                continue_while!(is_valid_base16_byte);
                let bytes = &input[input_index..=end_input_index];

                let bytes = (0..bytes.len())
                    .step_by(2)
                    .map(|i| u8::from_str_radix(&bytes[i..i + 2], 16))
                    .collect::<Result<Vec<u8>, _>>()
                    .map_err(|e| (ParseError::Int(e), input_index))?;

                // Create the decoded bytes string
                ast.bytes(bytes, input_index.try_into().unwrap())
            }
            TokenId::LeftBracket => {

                // Parse a list
                let mut nodes = Vec::new();
                let mut prev_token = TokenId::LeftBracket;

                if matches!(tokens[*token_index].id, TokenId::RightBracket) {
                    // Handle the empty list case
                    *token_index += 1;
                } else {
                    // Otherwise, parse all the tokens in the list

                    while !matches!(tokens[*token_index].id, TokenId::RightBracket) {
                        let next_token = tokens[*token_index];

                        if matches!(next_token.id, TokenId::EndOfFile) {
                            *token_index += 1;
                            break;
                        }

                        if matches!(next_token.id, TokenId::Comma) {
                            if matches!(prev_token, TokenId::LeftBracket | TokenId::Comma) {
                                    return Err((ParseError::ListCommaBeforeToken, next_token.pos as usize));
                            }

                            *token_index += 1;
                            continue;
                        }

                        let token = _parse(tokens, token_index, input, ast, 20)?;
                        nodes.push(token);


                        // dbg!(&ast.nodes[token]);

                        prev_token = next_token.id; 
                    }

                    if !matches!(tokens[*token_index].id, TokenId::RightBracket) {
                        return Err((ParseError::InternalParseError, tokens[*token_index].pos as usize));
                    }

                    // Skip over the parsed right bracket
                    *token_index += 1;
                }

                ast.list(nodes, input_index.try_into().unwrap())
            }
            TokenId::LeftParen => {
                let next_token = tokens[*token_index];
                if matches!(next_token.id, TokenId::RightParen) {
                    ast.hole(start.pos)
                } else {
                    let result =  _parse(tokens, token_index, input, ast, 0)?;


                    if matches!(tokens[*token_index].id, TokenId::RightParen) {
                        *token_index += 1;
                    }

                    result
                }
            }
            TokenId::LeftBrace => {
                let mut prev_token = TokenId::LeftBrace;

                let mut keys = Vec::new();
                let mut values = Vec::new();

                if matches!(tokens[*token_index].id, TokenId::RightBrace) {
                    // Nothing to do, return an empty record
                } else {
                    while !matches!(tokens[*token_index].id, TokenId::RightBrace) {
                        let next_token = tokens[*token_index];
                        dbg!(&next_token);

                        if matches!(next_token.id, TokenId::EndOfFile) {
                            *token_index += 1;
                            break;
                        }

                        if matches!(next_token.id, TokenId::Comma) {
                            if matches!(prev_token, TokenId::LeftBrace | TokenId::Comma) {
                                return Err((ParseError::RecordCommaBeforeAssignment, next_token.pos as usize));
                            }

                            *token_index += 1;
                            continue;
                        }

                        let old_token_index = *token_index;
                        let token = _parse(tokens, token_index, input, ast, 20)?;

                        match &ast.nodes[token] {
                            Node::Assign { left, right } => {
                                keys.push(*left);
                                values.push(*right);
                            }
                            Node::Spread { name: _ } => {
                                
                            }
                            x => {
                                dbg!(&x);
                                return Err((ParseError::NonAssignmentInRecord(tokens[old_token_index].id), next_token.pos as usize));
                            }
                        }

                        prev_token = next_token.id; 
                    }
            
                    // Skip over the right brace
                    *token_index += 1;
                }

                ast.record(keys, values, input_index.try_into().unwrap())
            }
            TokenId::EndOfFile | TokenId::RightBracket | TokenId::RightBrace | TokenId::RightParen => {
                return Err((ParseError::NonLeafNodeFoundFirst, start.pos as usize));
            }
            TokenId::OpMatchCase => {
                let mut matches = Vec::new();

                *token_index -= 1;

                while matches!(tokens[*token_index].id, TokenId::OpMatchCase) {
                    *token_index += 1;

                    let old_token_index = *token_index;
                    let next_token = tokens[*token_index];
                    let case= _parse(tokens, token_index, input, ast, 0)?;

                    match ast.nodes.get(case.0) {
                        Some(Node::Function { arg: name, body }) => {
                            // Replace the Function with a MatchCase node
                            ast.nodes[case] = Node::MatchCase { left: *name, right: *body };
                            matches.push(case);
                        }
                        _ => {
                            return Err((ParseError::NonFunctionInMatch(tokens[old_token_index].id), next_token.pos as usize));
                        }
                    }
                }

                ast.match_function(matches, start.pos)
            }
            TokenId::OpWhere => {
                todo!()
                
            }
            x => return Err((ParseError::UnknownToken(x), start.pos as usize)),
        };

        Ok(leaf)
    };

    let mut left = parse_leaf(ast, token_index)?;

    // dbg!(left, &ast.nodes[left]);

    loop {
        // ast.print_nodes(input);

        // Check if the left was modified
        let original = left;

        let Some(next) = tokens.get(*token_index) else {
            println!("Breaking out of the loop..");
            break;
        };

        if matches!(next.id, 
            TokenId::EndOfFile | TokenId::RightBracket | TokenId::RightBrace | TokenId::RightParen | TokenId::OpMatchCase) {

            // dbg!(next.id);
            break;
        }

        if let Some((binary_precedence, checked)) = next.id.get_op_precedence() {
            let binary_op = next;
            // Is a binary operator

            // If the next prescedence is less than the current, return out of the loop
            // There are two values given for precedence. 
            if checked < current_precedence {
                return Ok(left);
            }

            // Increment the token index
            *token_index += 1;

            let right = _parse(tokens, token_index, input, ast, binary_precedence)?;

            match binary_op.id {
                TokenId::EndOfFile => {
                    println!("Hit EOF");
                    break;
                }
                TokenId::OpAdd => {
                    left = ast.add(left, right, binary_op.pos);
                }
                TokenId::OpSub => {
                    left = ast.sub(left, right, binary_op.pos);
                }
                TokenId::OpMul => {
                    left = ast.mul(left, right, binary_op.pos);
                }
                TokenId::OpExp => {
                    left = ast.exp(left, right, binary_op.pos);
                }
                TokenId::OpGreater => {
                    left = ast.greater_than(left, right, binary_op.pos);
                }
                TokenId::OpGreaterEqual => {
                    left = ast.greater_equal(left, right, binary_op.pos);
                }
                TokenId::OpLess => {
                    left = ast.less_than(left, right, binary_op.pos);
                }
                TokenId::OpLessEqual => {
                    left = ast.less_equal(left, right, binary_op.pos);
                }
                TokenId::OpAccess => {
                    left = ast.access(left, right, binary_op.pos);
                }
                TokenId::OpListAppend => {
                    left = ast.list_append(left, right, binary_op.pos);
                }
                TokenId::OpStrConcat => {
                    left = ast.str_concat(left, right, binary_op.pos);
                }
                TokenId::OpListCons => {
                    left = ast.list_cons(left, right, binary_op.pos);
                }
                TokenId::OpDiv => {
                    left = ast.div(left, right, binary_op.pos);
                }
                TokenId::OpFloorDiv => {
                    left = ast.floor_div(left, right, binary_op.pos);
                }
                TokenId::OpMod => {
                    left = ast.modulo(left, right, binary_op.pos);
                }
                TokenId::OpEqual => {
                    left = ast.equal(left, right, binary_op.pos);
                }
                TokenId::OpNotEqual => {
                    left = ast.not_equal(left, right, binary_op.pos);
                }
                TokenId::OpAnd => {
                    left = ast.and(left, right, binary_op.pos);
                }
                TokenId::OpOr => {
                    left = ast.or(left, right, binary_op.pos);
                }
                TokenId::OpAssign => {
                    let left_node = &ast.nodes[left];

                    // Only allow variables on left side of assignments
                    if !matches!(left_node, Node::Var { .. }) {
                        assert!(*token_index >= 3,
                            "Less than 3 tokens before hitting assign? {}", *token_index);

                        let prev_token = tokens[*token_index - 3];
                        return Err((ParseError::NonVariableInAssignment, prev_token.pos as usize));
                    }

                    left = ast.assign(left, right, binary_op.pos);
                }
                TokenId::OpFunction => {
                    left = ast.function(left, right, binary_op.pos);
                }
                TokenId::OpWhere => {
                    left = ast.where_op(left, right, binary_op.pos);
                }
                TokenId::OpAssert => {
                    left = ast.assert(left, right, binary_op.pos);
                }
                TokenId::OpHasType => {
                    left = ast.has_type(left, right, binary_op.pos);
                }
                TokenId::OpPipe => {
                    left = ast.apply(right, left, binary_op.pos);
                }
                TokenId::OpReversePipe => {
                    left = ast.apply(left, right, binary_op.pos);
                }
                TokenId::OpRightEval => {
                    left = ast.right_eval(left, right, binary_op.pos);
                }
                TokenId::OpCompose => {
                    left = ast.compose(left, right, binary_op.pos);
                }
                TokenId::OpComposeReverse => {
                    left = ast.compose(right, left, binary_op.pos);
                }
                x => {
                    return Err((ParseError::UnknownOperatorToken(x), binary_op.pos as usize));
                }
            }
        } else {
            // Name followed by Name isn't a specific operation in tokens
            const OPAPPLY_PRECEDENCE: u32 = 1000;
            if OPAPPLY_PRECEDENCE < current_precedence {
                break;
            }

            let right = _parse(tokens, token_index, input, ast, OPAPPLY_PRECEDENCE + 2)?;


            if !matches!(ast.get_type(&left), Type::Apply | Type::Function | Type::Compose | Type::Var | Type::Sub) {
                return Err((ParseError::InvalidFunctionTypeInApply(ast.get_type(&left)), ast.positions[left.0] as usize));
            }

            left = ast.apply(left, right, next.pos - 1);
        }

        // Base case: The tree didn't move, we've reached the end
        if left == original {
            break;
        }
    }

    Ok(left)
}


/// Format an error message to be printed
#[allow(dead_code)]
fn print_error(input: &str, error: &Error) {
    const NEW_LINES_AROUND_ERROR: u32 = 2;
    
    let (error_str, location) = match error {
        Error::Tokenize((ref err, location)) => {
            (format!("{err:?}"), *location)
        }
        Error::Parse((ref err, location)) => {
            (format!("{err:?}"), *location)
        }
        Error::Eval((ref err, location)) => {
            (format!("{err:?}"), *location)
        }
    };

    println!("{}: {}",
        "error".to_string().bold_red(),
        error_str.to_string().bold_bright_white()
    );
    println!();

    // Find the start of the error block X new lines backwards
    let mut start = 0;
    let mut new_lines = 0;
    for i in (0..location).rev() {
        start = i;
        if input.chars().nth(i) == Some('\n') {
            new_lines += 1;
            if new_lines >= NEW_LINES_AROUND_ERROR {
                break;
            }
        }
    }

    // Find the end of the error block X new lines forwards
    let mut end= 0;
    let mut new_lines = 0;
    for i in location..input.len() {
        end = i;

        if input.chars().nth(i) == Some('\n') {
            new_lines += 1;
            if new_lines >= NEW_LINES_AROUND_ERROR {
                break;
            }
        }
    }

    // Find the line number the error was on
    let mut line_num = 0;
    let mut sum = 0;
    for (line_index, line) in input.lines().enumerate() {
        if (sum..sum + line.len()).contains(&location) {
            line_num = line_index;
            break;
        }

        sum += line.len();
    }

   
    let curr_input = &input[start..=end];
    let mut curr_loc = start;

    println!("{:6} {} input_file:line_num:col_um", 
        "",
        "/".to_string().bold_bright_blue(),
    );

    // If the error is early in the file, still add the padding above the
    // error for consistency
    if line_num < 2 {
        println!("{}", format!("{:6} | ", "").bold_bright_blue());
    }
    
    for line in curr_input.lines() {
        let has_error = (curr_loc..curr_loc + line.len()).contains(&location);

        let mut padding = String::new();
        if has_error {
            padding = format!("{line_num:6}");
        }

        println!("{}{line}", format!("{padding:6} | ").bold_bright_blue());

        // If this line has the error in it, print it here
        if has_error {
            let curr_input = input[location..].split('\n').next().unwrap();
            let tokens = tokenize(curr_input).unwrap();
            assert!(tokens.len() >= 2);
            let mut token_len = token_length(curr_input).unwrap();

            if token_len + 1 < input.len() {
                token_len += 1;
            }

            let token_marker = format!("{}{}", 
                " ".repeat(location - curr_loc),
                "^".repeat(token_len as usize).bold_yellow(),
            );

            // Create the marker line
            let mut token_line = format!("{:6} {} {token_marker} ",
                "", 
                "|".to_string().bold_bright_blue(),
            );

            // Add a help message if one applies for this error
            if let Some(help) = error.help() {
                token_line.push_str(&format!("help: {help}").bold_yellow().to_string());
            };

            println!("{token_line}");
        } 

        // Line length plus new line character
        curr_loc += line.len() + 1;
    }
    
    let mut note_line =  format!("{:6} {} ", 
        "",
        "\\".to_string().bold_bright_blue(),
    );

    if let Some(note) = error.note() {
        note_line.push_str(&format!("note: {note}").bold_white().to_string());
    }

    println!("{note_line}");

    println!();
}

/// Get the length of the token located at the beginning of the given string
///
/// # Errors
///
/// * Token parsing error along with the position where the error occured
#[allow(clippy::too_many_lines)]
pub fn token_length(input: &str) -> Result<usize, (TokenError, usize)> {
    let input = input.as_bytes();

    let mut index = 0;
    let mut pos;

    macro_rules! continue_while_space {
        () => {
            // Skip over all whitespace. Reset the current token when hitting whitespace.
            while matches!(input[index], b' ' | b'\n' | b'\r' | b'\t') {
                index += 1;
            }
        };
    }

    macro_rules! continue_while {
        ($until:pat) => {
            // Skip over all whitespace. Reset the current token when hitting whitespace.
            while matches!(input[index], $until) {
                index += 1;
            }
        };
    }

    macro_rules! continue_until {
        ($until:pat) => {
            // Skip over all whitespace. Reset the current token when hitting whitespace.
            while !matches!(input[index], $until) {
                index += 1;
            }

            // Skip over the found pattern
            index += 1;
        };
    }

    // Continue past the whitespace
    continue_while_space!();
    pos = index;

    // println!("{pos}: {}", input[index] as char);
    match &input {
        [b'-', b'-', ..] => {
            // Comments are -- COMMENT
            index += 2;
            continue_until!(b'\n');
        }
        [b'"', ..] => {
            index += 1;

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
            index += 3;
        }
        [b'.', b'.', ..] => {
            return Err((TokenError::DotDot, pos));
        }
        [b'~', b'~', b'8', b'5', b'\'', ..] => {
            pos += 5;
            index += 5;

            continue_while!(b'0'..=b'9' | b'A'..=b'Z' | b'a'..=b'z' | 
                b'!' | b'#' | b'$' | b'%' | b'&' | b'(' | b')' | b'*' | 
                b'+' | b'-' | b';' | b'<' | b'=' | b'>' | b'?' | b'@' | 
                b'^' | b'_' | b'`' | b'{' | b'|' | b'}' | b'~' | b'"');

            println!("Base85: After: {pos} {index}");
        }
        [b'~', b'~', b'3', b'2', b'\'', ..] => {
            pos += 5;
            index += 5;
            continue_while!(b'A'..=b'Z' | b'2'..=b'7' | b'=');
            println!("Base32: After: {pos} {index}");
        }
        [b'~', b'~', b'1', b'6', b'\'', ..] => {
            index += 5;
            continue_while!(b'A'..=b'F' | b'a'..=b'f' | b'0'..=b'9');
        }
        [b'~', b'~', b'6', b'4', b'\'', ..] => {
            index += 5;
            continue_while!(b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'=');
        }
        [b'~', b'~', ..] => {
            index += 2;
            continue_while!(b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'=');
        }
        #[allow(clippy::unnested_or_patterns)]
        [b'.', b' ', ..]
        | [b'-', b'>', ..] 
        | [b'>', b'>', ..]
        | [b'<', b'<', ..]
        | [b'=', b'=', ..]
        | [b'/', b'/', ..]
        | [b'/', b'=', ..]
        | [b'<', b'=', ..]
        | [b'>', b'=', ..]
        | [b'&', b'&', ..]
        | [b'|', b'|', ..]
        | [b'+', b'+', ..]
        | [b'>', b'+', ..]
        | [b'+', b'<', ..]
        | [b'|', b'>', ..]
        | [b'<', b'|', ..] => {
            index += 2;
        }
        [b'0'..=b'9' | b'.', ..] => {
            let mut id = TokenId::Int;

            // Read all digits and potential '.' for floats
            while index < input.len() && matches!(input[index], b'0'..=b'9' | b'.') {
                if input[index] == b'.' {
                    if id == TokenId::Float {
                        return Err((TokenError::FloatWithMultipleDecimalPoints, index));
                    }

                    id = TokenId::Float;
                };

                index += 1;
            }
        }
        // Name [a-zA-z$'_][a-zA-Z$'_0-9]*
        [b'a'..=b'z' | b'A'..=b'Z' | b'$' | b'\'' | b'_', ..] => {
            // Only allow quotes for names that start with $
            let allow_quotes = input[index] == b'$';

            index += 1;

            // Skip over all characters allowed in a name
            while index < input.len() && matches!(input[index], b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'$' | b'_' | b'\'')
            {
                // Explicitly disallow quotes in the variable
                if !allow_quotes && input[index] == b'\'' {
                    return Err((TokenError::QuoteInVariable, index));
                }

                index += 1;
            }
        }
        [b'#', ..] => {
            index += 1;

            let mut is_empty = true;
            continue_while_space!();

            pos = index;

            // Skip over all characters allowed in a name
            while index < input.len() && is_valid_symbol_byte(input[index]) {
                index += 1;
                is_empty = false;
            }

            if is_empty {
                if index >= input.len() {
                    return Err((TokenError::UnexpectedEndOfFile, pos));
                }

                return Err((TokenError::InvalidSymbol, pos));
            }
        }
        [b'*' | b'+' | b'/' | b'^' | b'!' | b'@' | b':' | b'%' | b'<' | b'>' |
         b'=' | b'|' | b'(' | b')' | b'[' | b']' | b',' | b'{' | b'}' | b'?' |
         b'-', 
         .. ] => {
            index += 1;
        }
        x => {
            return Err((TokenError::UnknownCharacter(x[0] as char), pos));
        }
    }

    Ok(index - 1)
}

fn print_ast(ast: &SyntaxTree) {
    for (i, node) in ast.nodes.iter().enumerate() {
        println!("{i}: {node:?}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use TokenId::*;

    #[allow(clippy::needless_pass_by_value)]
    fn __impl_eval_test_with_env(
            input: &'static str, 
            init_env: &[(std::string::String, NodeId)],
            wanted_result: EvalResult, 
            result_env: &[(std::string::String, NodeId)]) -> Result<(), (EvalError, u32)> {
        let (root, mut ast) = parse_str(input).unwrap();

        ast.dump_dot(root, "/tmp/dump").unwrap();

        let mut ctx = Context::default();
        for (key, value) in init_env {
            ctx.env.insert(key.to_string(), *value);
        }

        let curr_result = match eval(&mut ctx, &mut ast, root) {
            Ok(result) => result,
            Err((err, loc)) => {
                print_error(input, &Error::Eval((err.clone(), loc.try_into().unwrap())));
                return Err((err, loc));
            }
        };

        println!("--- AST ---");
        for (i, node) in ast.nodes.iter().enumerate() {
            println!("{i}: {node:?}");
        }

        // ast.dump_dot(curr_result, "/tmp/dump_after").unwrap();
        dbg!(&curr_result);

        assert_eq!(curr_result, wanted_result);

        let mut res_env = HashMap::new();
        for (key, value) in result_env {
            res_env.insert(key.to_string(), *value);
        }

        assert_eq!(ctx.env, res_env);

        Ok(())
    }

    #[allow(clippy::needless_pass_by_value)]
    fn end_to_end_test(input: &'static str,  wanted_result: EvalResult) -> Result<(), (EvalError, u32)> {
        let (root, mut ast) = parse_str(input).unwrap();

        ast.dump_dot(root, "/tmp/dump").unwrap();

        let mut ctx = Context::default();
        let curr_result = match eval(&mut ctx, &mut ast, root) {
            Ok(result) => result,
            Err((err, loc)) => {
                print_error(input, &Error::Eval((err.clone(), loc.try_into().unwrap())));
                return Err((err, loc));
            }
        };

        println!("--- AST ---");
        for (i, node) in ast.nodes.iter().enumerate() {
            println!("{i}: {node:?}");
        }

        // ast.dump_dot(curr_result, "/tmp/dump_after").unwrap();

        dbg!(&curr_result);

        assert_eq!(curr_result, wanted_result);

        Ok(())
    }

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
    #[allow(clippy::cast_possible_truncation)]
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
            ("&&", OpAnd),
            ("||", OpOr),
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
                Token::new(LeftBracket, 0),
                Token::new(RightBracket, 2),
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
                Token::new(LeftBracket, 0),
                Token::new(Int, 2),
                Token::new(Comma, 4),
                Token::new(Int, 6),
                Token::new(RightBracket, 8),
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
                Token::new(LeftBracket, 0),
                Token::new(Int, 1),
                Token::new(Comma, 2),
                Token::new(Int, 3),
                Token::new(RightBracket, 4),
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
                Token::new(LeftBrace, 0),
                Token::new(RightBrace, 1),
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
                Token::new(LeftBrace, 0),
                Token::new(RightBrace, 3),
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
                Token::new(LeftBrace, 0),
                Token::new(Name, 2),
                Token::new(OpAssign, 4),
                Token::new(Int, 6),
                Token::new(RightBrace, 8),
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
                Token::new(LeftBrace, 0),
                Token::new(Name, 2),
                Token::new(OpAssign, 4),
                Token::new(Int, 6),
                Token::new(Comma, 7),
                Token::new(Name, 9),
                Token::new(OpAssign, 11),
                Token::new(String, 13),
                Token::new(RightBrace, 17),
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
                Token::new(LeftBracket, 0),
                Token::new(OpSpread, 2),
                Token::new(RightBracket, 6),
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
                Token::new(LeftBracket, 0),
                Token::new(Int, 2),
                Token::new(Comma, 4),
                Token::new(OpSpread, 6),
                Token::new(RightBracket, 10),
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
                Token::new(LeftBracket, 0),
                Token::new(Int, 1),
                Token::new(Comma, 2),
                Token::new(OpSpread, 3),
                Token::new(RightBracket, 6),
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
                Token::new(LeftBracket, 0),
                Token::new(Int, 1),
                Token::new(Comma, 2),
                Token::new(OpSpread, 3),
                Token::new(Name, 6),
                Token::new(RightBracket, 10),
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
                Token::new(LeftBrace, 0),
                Token::new(OpSpread, 2),
                Token::new(RightBrace, 6),
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
                Token::new(LeftBrace, 0),
                Token::new(Name, 2),
                Token::new(OpAssign, 4),
                Token::new(Int, 6),
                Token::new(Comma, 7),
                Token::new(OpSpread, 9),
                Token::new(RightBrace, 13),
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
                Token::new(LeftBrace, 0),
                Token::new(Name, 1),
                Token::new(OpAssign, 2),
                Token::new(Int, 3),
                Token::new(Comma, 4),
                Token::new(OpSpread, 5),
                Token::new(RightBrace, 8),
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
        let tokens = parse(&[], "");

        assert!(tokens.is_err());
        assert!(matches!(tokens.err(), Some((ParseError::NoTokensGiven, 0))));
    }

    #[test]
    fn test_parse_digit_returns_int() {
        let input = "1";
        let (_root, ast) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![Node::Int { data: 1 }],
                positions: vec![0]
            }
        );
    }

    #[test]
    fn test_parse_digits_returns_int() {
        let input = "123";
        let (_root, ast) = parse_str(input).unwrap();
        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![Node::Int { data: 123 }],
                positions: vec![0]
            }
        );
    }

    #[test]
    fn test_parse_negative_int_returns_binary_sub_int() {
        let input = "-123";
        let (_root, ast) = parse_str(input).unwrap();

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
                ],
                positions: vec![u32::MAX, 1, 0]
            }
        );
    }

    #[test]
    fn test_parse_negative_var() {
        let input = "-x";
        let (_root, ast) = parse_str(input).unwrap();

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
                ],
                positions: vec![u32::MAX, 1, 0]
            }
        );
    }

    #[test]
    fn test_parse_negative_int_binds_tighter_than_plus() {
        let input = "-l+r";
        let (_root, ast) = parse_str(input).unwrap();

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
                ],
                positions: vec![u32::MAX, 1, 0, 3, 2]
            }
        );
    }

    #[test]
    fn test_parse_negative_int_binds_tighter_than_mul() {
        let input = "-l*r";
        let (_root, ast) = parse_str(input).unwrap();

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
                ],
                positions: vec![u32::MAX, 1, 0, 3, 2]
            }
        );
    }

    #[test]
    fn test_parse_negative_int_binds_tighter_than_access() {
        let input = "-l@r";
        let (_root, ast) = parse_str(input).unwrap();

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
                ],
                positions: vec![u32::MAX, 1, 0, 3, 2]
            }
        );
    }

    #[test]
    fn test_parse_negative_int_binds_tighter_than_apply() {
        let input = "-l r";
        let (_root, ast) = parse_str(input).unwrap();

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
                        func: NodeId(2),
                        arg: NodeId(3)
                    },
                ],
                positions: vec![u32::MAX, 1, 0, 3, 2]
            }
        );
    }

    #[test]
    fn test_parse_decimal_return_float() {
        let input = "3.42";
        let (_root, ast) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![Node::Float { data: 3.42 },],
                positions: vec![0]
            }
        );
    }
    #[test]
    fn test_parse_decimal_return_returns_binary_sub_float() {
        let input = "-3.42";
        let (_root, ast) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Int { data: 0 },
                    Node::Float { data: 3.42 },
                    Node::Sub {
                        left: NodeId(0),
                        right: NodeId(1)
                    }
                ],
                positions: vec![u32::MAX, 1, 0]
            }
        );
    }
    #[test]
    fn test_parse_var_returns_var() {
        let input = "abc_123";
        let (_root, ast) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![Node::Var {
                    data: "abc_123".to_string()
                }],
                positions: vec![0]
            }
        );
    }

    #[test]
    fn test_parse_sha_var_returns_var() {
        let input = "$sha1'abc";
        let (_root, ast) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![Node::Var {
                    data: "$sha1'abc".to_string()
                }],
                positions: vec![0]
            }
        );
    }

    #[test]
    fn test_parse_sha_var_quote_returns_var() {
        let input = "$sha1'abc";
        let (_root, ast) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![Node::Var {
                    data: "$sha1'abc".to_string()
                }],
                positions: vec![0]
            }
        );
    }

    #[test]
    fn test_parse_dollar_dollar_returns_var() {
        let input = "$$";
        let (_root, ast) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![Node::Var {
                    data: "$$".to_string()
                }],
                positions: vec![0]
            }
        );
    }

    #[test]
    fn test_parse_dollar_returns_var() {
        let input = "$";
        let (_root, ast) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![Node::Var {
                    data: "$".to_string()
                }],
                positions: vec![0]
            }
        );
    }
    #[test]
    fn test_parse_dollar_dollar_return_var() {
        let input = "$$bills";
        let (_root, ast) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![Node::Var {
                    data: "$$bills".to_string()
                },]
                ,positions: vec![0]
            }
        );
    }

    #[test]
    fn test_parse_bytes_returns_bytes_base85() {
        let input = "~~85'K|(_";
        let (_root, ast) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![Node::Bytes {
                    data: "ABC".as_bytes().to_vec()
                },]
                ,positions: vec![5]
            }
        );
    }

    #[test]
    fn test_parse_bytes_returns_bytes_base64() {
        let input = "~~64'QUJD";
        let (_root, ast) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![Node::Bytes {
                    data: "ABC".as_bytes().to_vec()
                }],
                positions: vec![5]
            }
        );
    }

    #[test]
    fn test_parse_binary_add_returns_add() {
        let input = "1+2";
        let (_root, ast) = parse_str(input).unwrap();

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
                ,positions: vec![0, 2, 1]
            }
        );
    }

    #[test]
    fn test_parse_binary_sub_returns_sub() {
        let input = "1-2";
        let (_root, ast) = parse_str(input).unwrap();

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
                ],
                positions: vec![0, 2, 1]
            }
        );
    }

    #[test]
    fn test_parse_binary_add_right_returns_add() {
        let input = "1+2+3";
        let (_root, ast) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Int { data: 1 },
                    Node::Int { data: 2 },
                    Node::Int { data: 3 },
                    Node::Add {
                        left: NodeId(1),
                        right: NodeId(2)
                    },
                    Node::Add {
                        left: NodeId(0),
                        right: NodeId(3)
                    }
                ],
                positions: vec![0,2,4,3,1]
            }
        );
    }

    #[test]
    fn test_parse_binary_add_right_returns_mul() {
        let input = "1+2*3";
        let (_root, ast) = parse_str(input).unwrap();

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
                ,positions: vec![0, 2, 4, 3, 1]
            }
        );
    }

    #[test]
    fn test_parse_binary_add_right_returns_mul_left() {
        let input = "1*2+3";
        let (_root, ast) = parse_str(input).unwrap();

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
                ,positions: vec![0, 2, 1, 4, 3]
            }
        );
    }

    #[test]
    fn test_exp_binds_tighter_than_mul_right() {
        let input = "5*2^3";
        let (_root, ast) = parse_str(input).unwrap();

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
                ],
                positions: vec![0,2,4,3,1]
            }
        );
    }

    #[test]
    fn test_list_access_binds_tighter_than_append() {
        let input = "a +< ls@0";
        let (_root, ast) = parse_str(input).unwrap();

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
                ,positions: vec![0,5,8,7,2]
            }
        );
    }

    #[test]
    fn test_parse_binary_str_concat() {
        let input = "abc ++ def";
        let (_root, ast) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Var{ data: "abc".to_string() },
                    Node::Var{ data: "def".to_string() },
                    Node::StrConcat { left: NodeId(0), right: NodeId(1) },
                ],
                positions: vec![0, 7, 4]
            }
        );
    }

    #[test]
    fn test_parse_binary_list_cons() {
        let input = "a >+ c";
        let (_root, ast) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Var{ data: "a".to_string() },
                    Node::Var{ data: "c".to_string() },
                    Node::ListCons { left: NodeId(0), right: NodeId(1) },
                ],
                positions: vec![0, 5, 2]
            }
        );
    }

    #[test]
    fn test_parse_binary_list_append() {
        let input = "a +< c";
        let (_root, ast) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Var{ data: "a".to_string() },
                    Node::Var{ data: "c".to_string() },
                    Node::ListAppend { left: NodeId(0), right: NodeId(1) },
                ],
                positions: vec![0, 5, 2]
            }
        );
    }

    #[test]
    fn test_parse_binary_op() {
        let ops = [
            "+", "-", "*", "/", "^", "%", "==", "/=", "<", ">", "<=", ">=", "&&", "||", "+<"
        ];

        for op in ops {
            let input = format!("a {op} c");
            let (_root, ast) = parse_str(&input).unwrap();

            let left = NodeId(0);
            let right = NodeId(1);

            let node = match op {
                "+" => Node::Add { left, right }, 
                "-" => Node::Sub{ left, right }, 
                "*" => Node::Mul { left, right }, 
                "/" => Node::Div { left, right }, 
                "^" => Node::Exp { left, right }, 
                "%" => Node::Mod { left, right }, 
                "==" => Node::Equal { left, right }, 
                "/=" => Node::NotEqual { left, right }, 
                "<" => Node::LessThan { left, right }, 
                ">" => Node::GreaterThan { left, right },
                "<=" => Node::LessEqual { left, right }, 
                ">=" => Node::GreaterEqual { left, right },
                "&&" => Node::And { left, right }, 
                "||" => Node::Or { left, right }, 
                ">+" => Node::ListCons { left, right }, 
                "+<" => Node::ListAppend { left, right },
                _ => unreachable!()
            };

            assert_eq!(
                ast,
                SyntaxTree {
                    nodes: vec![
                        Node::Var{ data: "a".to_string() },
                        Node::Var{ data: "c".to_string() },
                        node
                    ],

                    #[allow(clippy::cast_possible_truncation)]
                    positions: vec![0, 3 + op.len() as u32, 2]
                }
            );
        }
    }

    #[test]
    fn test_parse_str_concat() {
        let input = "\"hello\" ++ \"world\"";
        let (_root, ast) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::String { data: "hello".to_string() },
                    Node::String { data: "world".to_string() },
                    Node::StrConcat { left: NodeId(0), right: NodeId(1) },
                ],
                positions: vec![0, 11, 8]
            }
        );
    }

    #[test]
    fn test_parse_empty_list() {
        let input = "[]";
        let (_root, ast) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::List { data: vec![] },
                ]
                ,positions: vec![0]
            }
        );
    }

    #[test]
    fn test_parse_empty_list_with_spaces() {
        let input = "[ ]";
        let (_root, ast) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::List { data: vec![] },
                ],
                positions: vec![0]
            }
        );
    }

    #[test]
    fn test_parse_list_with_items() {
        let input = "[ 1 , 2 ]";
        let (_root, ast) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Int { data: 1 },
                    Node::Int { data: 2 },
                    Node::List { data: vec![ NodeId(0), NodeId(1) ] },
                ],
                positions: vec![2, 6, 0]
            }
        );
    }

    #[test]
    fn test_parse_list_with_items_no_spaces() {
        let input = "[1*3, 2+4, [a,b]]";
        let (_root, ast) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Int { data: 1 },
                    Node::Int { data: 3 },
                    Node::Mul { left: NodeId(0), right: NodeId(1) },
                    Node::Int { data: 2 },
                    Node::Int { data: 4 },
                    Node::Add { left: NodeId(3), right: NodeId(4) },
                    Node::Var { data: "a".to_string() },
                    Node::Var { data: "b".to_string() },
                    Node::List { data: vec![ NodeId(6), NodeId(7) ] },
                    Node::List { data: vec![ NodeId(2), NodeId(5), NodeId(8) ] },
                ]
                ,positions: vec![1, 3, 2, 6, 8, 7, 12, 14, 11, 0]
            }
        );
    }

    #[test]
    fn test_parse_nested_lists() {
        let input = "[1*3, 2+4, [a,b]]";
        let (_root, ast) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Int { data: 1 },
                    Node::Int { data: 3 },
                    Node::Mul { left: NodeId(0), right: NodeId(1) },
                    Node::Int { data: 2 },
                    Node::Int { data: 4 },
                    Node::Add { left: NodeId(3), right: NodeId(4) },
                    Node::Var { data: "a".to_string() },
                    Node::Var { data: "b".to_string() },
                    Node::List { data: vec![ NodeId(6), NodeId(7) ] },
                    Node::List { data: vec![ NodeId(2), NodeId(5), NodeId(8) ] },
                ],
                positions: vec![1, 3, 2, 6, 8, 7, 12, 14, 11, 0]
            }
        );
    }

    #[test]
    fn test_pares_list_only_comma() {
        let input = "[,]";
        let output= parse_str(input);

        assert!(matches!(
            output,
            Err(Error::Parse((ParseError::ListCommaBeforeToken, 1)))
        ));

    }

    #[test]
    fn test_pares_list_only_two_commas() {
        let input = "[,,]";
        let output= parse_str(input);

        assert!(matches!(
            output,
            Err(Error::Parse((ParseError::ListCommaBeforeToken, 1)))
        ));

    }

    #[test]
    fn test_parse_assign() {
        let input = "a = 1";
        let (_root, ast ) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Var { data: "a".to_string() },
                    Node::Int { data: 1 },
                    Node::Assign { left: NodeId(0), right: NodeId(1) },
                ],
                positions: vec![0,4,2]
            }
        );
    }

    #[test]
    fn test_parse_function_one_arg() {
        let input = "a -> a + 1";
        let (_root, ast ) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Var { data: "a".to_string() },
                    Node::Var { data: "a".to_string() },
                    Node::Int { data: 1 },
                    Node::Add { left: NodeId(1), right: NodeId(2) },
                    Node::Function { arg: NodeId(0), body: NodeId(3) },
                ],
                positions: vec![0, 5, 9, 7, 2]
            }
        );
    }

    #[test]
    fn test_parse_function_two_args() {
        let input = "a -> b -> a + b";
        let (_root, ast ) = parse_str(input).unwrap();


        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Var { data: "a".to_string() },
                    Node::Var { data: "b".to_string() },
                    Node::Var { data: "a".to_string() },
                    Node::Var { data: "b".to_string() },
                    Node::Add { left: NodeId(2), right: NodeId(3) },
                    Node::Function { arg: NodeId(1), body: NodeId(4) },
                    Node::Function { arg: NodeId(0), body: NodeId(5) },
                ],
                positions: vec![0, 5, 10, 14, 12, 7, 2]
            }
        );
    }

    #[test]
    fn test_parse_assign_function() {
        let input = "id = x -> x";
        let (_root, ast ) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Var { data: "id".to_string() },
                    Node::Var { data: "x".to_string() },
                    Node::Var { data: "x".to_string() },
                    Node::Function { arg: NodeId(1), body: NodeId(2) },
                    Node::Assign{ left: NodeId(0), right: NodeId(3) },
                ],
                positions: vec![0, 5, 10, 7, 3]
            }
        );
    }

    #[test]
    fn test_function_application_one_arg() {
        let input = "f a";
        let (_root, ast ) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Var { data: "f".to_string() },
                    Node::Var { data: "a".to_string() },
                    Node::Apply { func: NodeId(0), arg: NodeId(1) },
                ],
                positions: vec![0,2,1]
            }
        );
    }

    #[test]
    fn test_parse_function_application_two_args() {
        let input = "f a b";
        let (_root, ast ) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Var { data: "f".to_string() },
                    Node::Var { data: "a".to_string() },
                    Node::Apply { func: NodeId(0), arg: NodeId(1) },
                    Node::Var { data: "b".to_string() },
                    Node::Apply { func: NodeId(2), arg: NodeId(3) },
                ],
                positions: vec![0,2,1,4,3]
            }
        );
    }

    #[test]
    fn test_parse_where() {
        let input = "a . b";
        let (_root, ast ) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Var { data: "a".to_string() },
                    Node::Var { data: "b".to_string() },
                    Node::Where { left: NodeId(0), right: NodeId(1) },
                ],
                positions: vec![0, 4, 2]
            }
        );
    }

    #[test]
    fn test_parse_nested_where() {
        let input = "a . b . c";
        let (_root, ast ) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Var { data: "a".to_string() },
                    Node::Var { data: "b".to_string() },
                    Node::Where { left: NodeId(0), right: NodeId(1) },
                    Node::Var { data: "c".to_string() },
                    Node::Where { left: NodeId(2), right: NodeId(3) },
                ],
                positions: vec![0, 4, 2, 8, 6]
            }
        );
    }

    #[test]
    fn test_parse_assert() {
        let input = "a ? b";
        let (_root, ast ) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Var { data: "a".to_string() },
                    Node::Var { data: "b".to_string() },
                    Node::Assert { left: NodeId(0), right: NodeId(1) },
                ]
                ,positions: vec![0,4,2]
            }
        );
    }

    #[test]
    fn test_parse_nested_assert() {
        let input = "a ? b ? c";
        let (_root, ast ) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Var { data: "a".to_string() },
                    Node::Var { data: "b".to_string() },
                    Node::Assert { left: NodeId(0), right: NodeId(1) },
                    Node::Var { data: "c".to_string() },
                    Node::Assert { left: NodeId(2), right: NodeId(3) },
                ],
                positions: vec![0, 4, 2, 8, 6]
            }
        );
    }

    #[test]
    fn test_parse_hastype() {
        let input = "a : b";
        let (_root, ast ) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Var { data: "a".to_string() },
                    Node::Var { data: "b".to_string() },
                    Node::HasType { left: NodeId(0), right: NodeId(1) },
                ],
                positions: vec![0, 4, 2]
            }
        );
    }

    #[test]
    fn test_parse_hole() {
        let input = "()";
        let (_root, ast ) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Hole,
                ],
                positions: vec![0]
            }
        );
    }

    #[test]
    fn test_parse_parenthesized_expression() {
        let input = "(1+2)";
        let (_root, ast ) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Int { data: 1 },
                    Node::Int { data: 2 },
                    Node::Add{ left: NodeId(0), right: NodeId(1) },
                ],
                positions: vec![1, 3, 2]
            }
        );
    }

    #[test]
    fn test_parse_parenthesized_expression_add_mul() {
        let input = "(1+2)*3";
        let (_root, ast ) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Int { data: 1 },
                    Node::Int { data: 2 },
                    Node::Add { left: NodeId(0), right: NodeId(1) },
                    Node::Int { data: 3 },
                    Node::Mul { left: NodeId(2), right: NodeId(3) },
                ],
                positions: vec![1, 3, 2, 6, 5]
            }
        );
    }

    #[test]
    fn test_parse_pipe() {
        let input = "1 |> f";
        let (_root, ast ) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Int { data: 1 },
                    Node::Var { data: "f".to_string() },
                    Node::Apply { func: NodeId(1), arg: NodeId(0) },
                ],
                positions: vec![0, 5, 2]
            }
        );
    }

    #[test]
    fn test_parse_nested_pipe() {
        let input = "1 |> f |> g";
        let (_root, ast ) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Int { data: 1 },
                    Node::Var { data: "f".to_string() },
                    Node::Apply { func: NodeId(1), arg: NodeId(0) },
                    Node::Var { data: "g".to_string() },
                    Node::Apply { func: NodeId(3), arg: NodeId(2) },
                ]
                ,positions: vec![0, 5, 2, 10, 7]
            }
        );
    }

    #[test]
    fn test_parse_reverse_pipe() {
        let input = "f <| 1";
        let (_root, ast ) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Var { data: "f".to_string() },
                    Node::Int { data: 1 },
                    Node::Apply { func: NodeId(0), arg: NodeId(1) },
                ],
                positions: vec![0, 5, 2]
            }
        );
    }

    #[test]
    fn test_parse_nested_reverse_pipe() {
        let input = "g <| f <| 1";
        let (_root, ast ) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Var { data: "g".to_string() },
                    Node::Var { data: "f".to_string() },
                    Node::Int { data: 1 },
                    Node::Apply { func: NodeId(1), arg: NodeId(2) },
                    Node::Apply { func: NodeId(0), arg: NodeId(3) },
                ],
                positions: vec![0, 5, 10, 7, 2]
            }
        );
    }

    #[test]
    fn test_parse_empty_record() {
        let input = "{}";
        let (_root, ast ) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Record { keys: Vec::new(), values: Vec::new() },
               ],
               positions: vec![0]
            }
        );
    }

    #[test]
    fn test_parse_empty_single_field() {
        let input = "{ a=4 }";
        let (_root, ast ) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Var { data: "a".to_string() },
                    Node::Int { data: 4 },
                    Node::Assign { left: NodeId(0), right: NodeId(1) },
                    Node::Record { 
                        keys: vec![NodeId(0)], 
                        values: vec![NodeId(1)]
                    },
                ],
                positions: vec![2, 4, 3, 0]
            }
        );
    }

    #[test]
    fn test_parse_empty_with_expression() {
        let input = "{ a=1+2 }";
        let (_root, ast ) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Var { data: "a".to_string() },
                    Node::Int { data: 1 },
                    Node::Int { data: 2 },
                    Node::Add { left: NodeId(1), right: NodeId(2) },
                    Node::Assign { left: NodeId(0), right: NodeId(3) },
                    Node::Record { 
                        keys: vec![NodeId(0)], 
                        values: vec![NodeId(3)]
                    },
                ],
                positions: vec![2, 4, 6, 5, 3, 0]
            }
        );
    }

    #[test]
    fn test_parse_empty_multiple_fields() {
        let input = "{ a=4, b=z }";
        let (_root, ast ) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Var { data: "a".to_string() },
                    Node::Int { data: 4 },
                    Node::Assign { left: NodeId(0), right: NodeId(1) },
                    Node::Var { data: "b".to_string() },
                    Node::Var { data: "z".to_string() },
                    Node::Assign { left: NodeId(3), right: NodeId(4) },
                    Node::Record { 
                        keys: vec![NodeId(0), NodeId(3)], 
                        values: vec![NodeId(1), NodeId(4)]
                    },
                ],
                positions: vec![2, 4, 3, 7, 9, 8, 0]
            }
        );
    }

    #[test]
    fn test_non_variable_in_assignment_raises_parse_error() {
        let input = "3=4";
        let output = parse_str(input);

        dbg!(&output);

        assert!(matches!(
            output,
            Err(Error::Parse((ParseError::NonVariableInAssignment, 0)))
        ));

    }

    #[test]
    fn test_non_assign_in_record_constructor_raises_parse_error() {
        let input = "{ 1,2 }";
        let output = parse_str(input);

        dbg!(&output);

        assert!(matches!(
            output,
            Err(Error::Parse((ParseError::NonAssignmentInRecord { .. }, 2)))
        ));

    }

    #[test]
    fn test_parse_right_eval() {
        let input = "a!b";
        let (_root, ast ) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Var { data: "a".to_string() },
                    Node::Var { data: "b".to_string() },
                    Node::RightEval{ left: NodeId(0), right: NodeId(1) },
                ],
                positions: vec![0, 2, 1]
            }
        );
    }

    #[test]
    fn test_parse_right_eval_with_defs() {
        let input = "a ! b . c";
        let (_root, ast ) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Var { data: "a".to_string() },
                    Node::Var { data: "b".to_string() },
                    Node::Var { data: "c".to_string() },
                    Node::Where{ left: NodeId(1), right: NodeId(2) },
                    Node::RightEval{ left: NodeId(0), right: NodeId(3) },
                ],
                positions: vec![0, 4, 8, 6, 2]
            }
        );
    }

    #[test]
    fn test_parse_match_no_cases_raises_parse_error() {
        let input = "|";
        let output= parse_str(input);

        assert!(matches!(
            output,
            Err(Error::Parse((ParseError::UnexpectedEndOfFile, 1)))
        ));
    }

    #[test]
    fn test_parse_match_one_case() {
        let input = "| 1 -> 2";
        let (_root, ast ) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Int { data: 1 },
                    Node::Int { data: 2 },
                    Node::MatchCase { left: NodeId(0), right: NodeId(1) },
                    Node::MatchFunction { data: vec![NodeId(2)]},
                ],
                positions: vec![2, 7, 4, 0]
            }
        );
    }

    #[test]
    fn test_parse_match_three_cases() {
        let input = r"
            | 1 -> a
            | 2 -> b
            | 3 -> c
        ";
        let (_root, ast ) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Int { data: 1 },
                    Node::Var { data: "a".to_string() },
                    Node::MatchCase { left: NodeId(0), right: NodeId(1) },
                    Node::Int { data: 2 },
                    Node::Var { data: "b".to_string() },
                    Node::MatchCase { left: NodeId(3), right: NodeId(4) },
                    Node::Int { data: 3 },
                    Node::Var { data: "c".to_string() },
                    Node::MatchCase { left: NodeId(6), right: NodeId(7) },
                    Node::MatchFunction { data: vec![
                        NodeId(2),
                        NodeId(5),
                        NodeId(8),
                    ]},
                ],
                positions: vec![15, 20, 17, 36, 41, 38, 57, 62, 59, 13]
            }
        );
    }

    #[test]
    fn test_parse_compose() {
        let input = r"f >> g";
        let (_root, ast ) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Var { data: "f".to_string() },
                    Node::Var { data: "g".to_string() },
                    Node::Compose { left: NodeId(0), right: NodeId(1) }
                ]
                ,positions: vec![0, 5, 2]
            }
        );
    }

    #[test]
    fn test_parse_compose_reverse() {
        let input = r"f << g";
        let (_root, ast ) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Var { data: "f".to_string() },
                    Node::Var { data: "g".to_string() },
                    Node::Compose { left: NodeId(1), right: NodeId(0) }
                ],
                positions: vec![0, 5, 2]
            }
        );
    }

    #[test]
    fn test_parse_double_compose() {
        let input = r"f << g << h";
        let (_root, ast ) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Var { data: "f".to_string() },
                    Node::Var { data: "g".to_string() },
                    Node::Var { data: "h".to_string() },
                    Node::Compose { left: NodeId(2), right: NodeId(1) },
                    Node::Compose { left: NodeId(3), right: NodeId(0) }
                ],
                positions: vec![0, 5, 10, 7, 2]
            }
        );
    }

    #[test]
    fn test_parse_and_tighter_than_or() {
        let input = r"x || y && z";
        let (_root, ast ) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Var { data: "x".to_string() },
                    Node::Var { data: "y".to_string() },
                    Node::Var { data: "z".to_string() },
                    Node::And { left: NodeId(1), right: NodeId(2) },
                    Node::Or { left: NodeId(0), right: NodeId(3) }
                ]
                ,positions: vec![0,5,10,7,2]
            }
        );
    }

    #[test]
    fn test_parse_list_spread() {
        let input = r"[1, ... ]";
        let (_root, ast ) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Int { data: 1 },
                    Node::Spread { name: None },
                    Node::List { data: vec![ NodeId(0), NodeId(1) ]}
                ],
                positions: vec![1, 4, 0]
            }
        );
    }

    #[test]
    fn test_parse_list_with_non_name_expr_after_spread_raises_parse_error() {
        let input = r"
            [1, ...rest, 2]
        ";

        let output = parse_str(input);

        assert!(matches!(
            output,
            Err(Error::Parse((ParseError::SpreadMustComeAtEndOfCollection, 1)))
        ));

    }

    #[test]
    fn test_parse_list_with_non_name_expr_after_spread_raises_parse_err() {
        let input = r"{x=1, ...}";

        let (_root, ast) = parse_str(input).unwrap();
        // let _ = ast.dump_dot(root, "/tmp/dump");

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Var { data: "x".to_string() },
                    Node::Int { data: 1 },
                    Node::Assign { left: NodeId(0), right: NodeId(1) },
                    Node::Spread { name: None },
                    Node::Record { keys: vec![NodeId(0)], values: vec![NodeId(1)] }
                ],
                positions: vec![1, 3, 2, 6, 0]
            }
        );

    }

    #[test]
    fn test_parse_record_spread_beginning_raises_parse_error() {
        let input = r"{..., 1, 2, 3}";

        let output = parse_str(input);

        dbg!(&output);

        assert!(matches!(
            output,
            Err(Error::Parse((ParseError::SpreadMustComeAtEndOfCollection, 1)))
        ));

    }

    #[test]
    fn test_parse_list_spread_beginning_raises_parse_error() {
        let input = r"[..., 1, 2, 3]";

        let output = parse_str(input);

        dbg!(&output);

        assert!(matches!(
            output,
            Err(Error::Parse((ParseError::SpreadMustComeAtEndOfCollection, 1)))
        ));

    }

    #[test]
    fn test_parse_list_spread_middle_raises_parse_error() {
        let input = r"[1, 2, ..., 3]";

        let output = parse_str(input);

        dbg!(&output);

        assert!(matches!(
            output,
            Err(Error::Parse((ParseError::SpreadMustComeAtEndOfCollection, 7)))
        ));

    }

    #[test]
    fn test_parse_record_spread_middle_raises_parse_error() {
        let input = r"{x=1, ..., y=2}";

        let output = parse_str(input);

        dbg!(&output);

        assert!(matches!(
            output,
            Err(Error::Parse((ParseError::SpreadMustComeAtEndOfCollection, 6)))
        ));

    }

    #[test]
    fn test_parse_record_with_only_comma_parse_error() {
        let input = r"{,}";

        let output = parse_str(input);

        dbg!(&output);

        assert!(matches!(
            output,
            Err(Error::Parse((ParseError::RecordCommaBeforeAssignment, 1)))
        ));

    }

    #[test]
    fn test_parse_record_with_two_comma_parse_error() {
        let input = r"{,,}";

        let output = parse_str(input);

        dbg!(&output);

        assert!(matches!(
            output,
            Err(Error::Parse((ParseError::RecordCommaBeforeAssignment, 1)))
        ));

    }

    #[test]
    fn test_parse_symbol() {
        let input = r"#abc";
        let (_root, ast) = parse_str(input).unwrap();

        assert_eq!(
            ast,
            SyntaxTree {
                nodes: vec![
                    Node::Symbol { data: "abc".to_string() },
                ],
                positions: vec![0]
            }
        );

    }

    /////////////////////////////////////////////////////////////////
    // Eval tests
    /////////////////////////////////////////////////////////////////
    
    #[test]
    fn test_eval_int() {
        end_to_end_test("1", 1.into()).unwrap();
    }

    #[test]
    fn test_eval_float() {
        end_to_end_test("2.21", 2.21.into()).unwrap();
    }

    #[test]
    fn test_eval_add() {
        end_to_end_test("1+2", 3.into()).unwrap();
    }

    #[test]
    fn test_eval_nested_add() {
        end_to_end_test("1+2+3", 6.into()).unwrap();
    }

    #[test]
    fn test_eval_nested_add_float() {
        end_to_end_test("-1.1+2.2+3.3-1.0", 3.4.into()).unwrap();
    }

    #[test]
    fn test_eval_sub() {
        end_to_end_test("1 - 2", (-1).into()).unwrap();
    }

    #[test]
    fn test_eval_sub_string() {
        let err = end_to_end_test("1 - hello", 1.into()).err().unwrap();
        assert_eq!(err, (EvalError::VariableNotFoundInScope("hello".to_string()), 4));
    }

    #[test]
    fn test_eval_mul() {
        end_to_end_test("3 * 2", 6.into()).unwrap();
    }

    #[test]
    fn test_eval_mul_float() {
        end_to_end_test("3.1 * 1.0", 3.1.into()).unwrap();
    }

    #[test]
    fn test_eval_mul_string() {
        let err = end_to_end_test("1 * hello", 1.into()).err().unwrap();
        assert_eq!(err,  (EvalError::VariableNotFoundInScope("hello".to_string()), 4));
    }

    #[test]
    fn test_eval_div() {
        end_to_end_test("8 / 2", 4.into()).unwrap();
    }

    #[test]
    fn test_eval_floor_div() {
        end_to_end_test("8.4 // 2.0", 4.0.into()).unwrap();
    }

    #[test]
    fn test_eval_exp() {
        end_to_end_test("2 ^ 3", 8.into()).unwrap();
    }

    #[test]
    fn test_eval_exp_floor() {
        end_to_end_test("2.0 ^ 3.0", 8.0.into()).unwrap();
    }

    #[test]
    fn test_eval_exp_floor_int_vs_float() {
        let err = end_to_end_test("2.0 ^ 3", 1.into()).err().unwrap();
        assert_eq!(err,  (EvalError::InvalidNodeTypes(Type::Exp), 4));
    }

    #[test]
    fn test_eval_mod() {
        end_to_end_test("8 % 3", 2.into()).unwrap();
    }

    #[test]
    fn test_eval_mod_floor() {
        end_to_end_test("8.0 % 2.5", 0.5.into()).unwrap();
    }

    #[test]
    fn test_eval_mod_floor_int_vs_float() {
        let err = end_to_end_test("8 % 2.0", 1.into()).err().unwrap();
        assert_eq!(err, (EvalError::InvalidNodeTypes(Type::Mod), 2));
    }

    #[test]
    fn test_eval_equal_int_true() {
        end_to_end_test("8 == 3", EvalResult::False).unwrap();
        end_to_end_test("8 /= 3", EvalResult::True).unwrap();
    }

    #[test]
    fn test_eval_equal_int_false() {
        end_to_end_test("8 == 8", EvalResult::True).unwrap();
        end_to_end_test("8 /= 8", EvalResult::False).unwrap();
    }

    #[test]
    fn test_eval_equal_float_true() {
        end_to_end_test("8.0 == 8.0", EvalResult::True).unwrap();
        end_to_end_test("8.0 /= 8.0", EvalResult::False).unwrap();
    }

    #[test]
    fn test_eval_equal_float_false() {
        end_to_end_test("8.0 == 2.0", EvalResult::False).unwrap();
        end_to_end_test("8.0 /= 2.0", EvalResult::True).unwrap();
    }

    #[test]
    fn test_eval_equal_floor_int_vs_float() {
        let err = end_to_end_test("8 == 2.0", 1.into()).err().unwrap();
        assert_eq!(err,  (EvalError::InvalidNodeTypes(Type::Equal), 2));

        let err = end_to_end_test("8 /= 2.0", 1.into()).err().unwrap();
        assert_eq!(err,  (EvalError::InvalidNodeTypes(Type::NotEqual), 2));
    }

    #[test]
    fn test_eval_equal_negative_float_false() {
        end_to_end_test("8.0 == -2.0", EvalResult::False).unwrap();
        end_to_end_test("8.0 /= -2.0", EvalResult::True).unwrap();
    }

    #[test]
    fn test_eval_equal_negative_float_true() {
        end_to_end_test("-8.0 == -8.0", EvalResult::True).unwrap();
        end_to_end_test("-8.0 /= -8.0", EvalResult::False).unwrap();
    }

    #[test]
    fn test_eval_equal_float_true_tests() {
        end_to_end_test("0.0 == 0.0", EvalResult::True).unwrap();
        end_to_end_test("0.00000001 == 0.0", EvalResult::False).unwrap();
        end_to_end_test("0.00000000001 == 0.0", EvalResult::False).unwrap();
        end_to_end_test("0.000000000000001 == 0.0", EvalResult::True).unwrap();

        end_to_end_test("0.0 /= 0.0", EvalResult::False).unwrap();
        end_to_end_test("0.00000001 /= 0.0", EvalResult::True).unwrap();
        end_to_end_test("0.00000000001 /= 0.0", EvalResult::True).unwrap();
        end_to_end_test("0.000000000000001 /= 0.0", EvalResult::False).unwrap();
    }

    #[test]
    fn test_eval_string_concat() {
        end_to_end_test("\"hello\" ++ \"world\"", "helloworld".into()).unwrap();
    }

    #[test]
    fn test_eval_string_concat_with_space() {
        end_to_end_test("\"hello\" ++ \" \" ++ \"world\"", "hello world".into()).unwrap();
    }

    #[test]
    fn test_eval_string_concat_error() {
        let err = end_to_end_test("\"hello\" ++ 1", 1.into()).err().unwrap();
        assert_eq!(err, (EvalError::NonStringFoundInStringConcat(Node::StrConcat { left: NodeId(0), right: NodeId(1) }), 8));
    }

    #[test]
    fn test_eval_string_concat_error2() {
        let err = end_to_end_test("1 ++ \"hello\"", 1.into()).err().unwrap();
        assert_eq!(err, (EvalError::NonStringFoundInStringConcat(Node::StrConcat { left: NodeId(0), right: NodeId(1) }), 2));
    }

    #[test]
    fn test_eval_list_cons() {
        end_to_end_test("1 >+ [2, 3]", 
            vec![ 1.into(), 2.into(), 3.into() ].into()
        ).unwrap();
    }

    #[test]
    fn test_eval_list_cons2() {
        let err = end_to_end_test("[2, 3] >+ 1", 1.into()).err().unwrap();
        assert_eq!(err, (EvalError::ListTypeMismatch(Type::Int), 10));
    }

    #[test]
    fn test_eval_list_cons3() {
        end_to_end_test("[] >+ []", 
            vec![
                vec![].into()
            ].into()
        ).unwrap();
    }

    #[test]
    fn test_eval_list_append() {
        end_to_end_test("[1, 2] +< 3", 
            vec![
                1.into(),
                2.into(),
                3.into()
            ].into()
        ).unwrap();
    }

    #[test]
    fn test_eval_list_append_evaluate_elements() {
        end_to_end_test("[1 + 2, 3 + 4]", 
            vec![ 3.into(), 7.into() ].into()
        ).unwrap();
    }

    #[test]
    fn test_eval_nested_where() {
        end_to_end_test(
            "
            a + b 
            . a = 1 
            . b = 2",
            3.into(),
        ).unwrap();
    }

    #[test]
    fn test_eval_assert_with_truthy_cond_returns_true() {
        end_to_end_test(
            "123 ? #true",
            123.into(),
        ).unwrap();
    }

    #[test]
    fn test_eval_assert_with_truthy_cond_returns_true_right_side() {
        end_to_end_test(
            "#true ? 123",
            123.into(),
        ).unwrap();
    }

    #[test]
    fn test_eval_assert_with_truthy_cond_returns_false() {
        let err = end_to_end_test(
            "123 ? #false",
            123.into(),
        ).err().unwrap();

        assert_eq!(err, (EvalError::FalseConditionFound, 6));
    }

    #[test]
    fn test_eval_nested_assert() {
        end_to_end_test(
            "123 ? #true ? #true",
            123.into(),
        ).unwrap();
    }

    #[test]
    fn test_eval_function_application_one_arg() {
        end_to_end_test(
            "(x -> x + 1) 2",
            3.into(),
        ).unwrap();
    }


    #[test]
    fn test_eval_record_access() {
        end_to_end_test(
            "
            rec@key2
            . rec = { a=1+4, key2=\"x\" ++ \"yy\"}",
            "xyy".into(),
        ).unwrap();
    }

    #[test]
    fn test_eval_record_access_with_invalid_key() {
        let err = end_to_end_test(
            "
            rec@badkey
            . rec = { a=1, key2=\"x\"}",
            "124".into(),
        );

        assert_eq!(
            err, 
            Err((EvalError::KeyNotFoundInRecord("badkey".to_string()), 17))
        );
    }

    #[test]
    fn test_eval_list_access() {
        end_to_end_test(
            "
            list@2
            . list = [1, 2, 3]",
            3.into(),
        ).unwrap();
    }

    #[test]
    fn test_eval_list_access_with_strings() {
        end_to_end_test(
            "
            list@1
            . list = [\"a\", \"b\", \"c\"]",
            "b".into(),
        ).unwrap();
    }

    #[test]
    fn test_eval_list_invalid_access_returns_error() {
        let err = end_to_end_test(
            "
            list@key2
            . list = [1, 2, 3]",
            "x".into(),
        );

        assert_eq!(
            err, 
            Err((EvalError::InvalidListAccessType(Type::Var), 18))
        );
    }

    #[test]
    fn test_eval_list_index_out_of_bounds() {
        let err = end_to_end_test(
            "
            list@999
            . list = [1, 2, 3]",
            "x".into(),
        );

        assert_eq!(
            err, 
            Err((EvalError::ListIndexOutOfBounds { length: 3, wanted_index: 999 }, 18))
        );
    }

    #[test]
    fn test_match_int_with_equal_int_matches() {
        end_to_end_test(
            "
            func 1 
            . func = | 1 -> 2",
            2.into(),
        ).unwrap();
    }

    #[test]
    fn test_match_int_with_equal_int_fail_matches_return_error() {
        let err = end_to_end_test(
            "
            func 999
            . func = | 1 -> 2",
            2.into(),
        );

        assert_eq!(
            err, 
            Err((EvalError::MatchCaseNotFound(Node::Int { data: 999 }, vec![Node::Int { data: 1 }]), 18))
        );
    }

    #[test]
    fn test_match_int_with_multiple_cases() {
        end_to_end_test(
            "
            func 2
            . func = 
                | 1 -> 2
                | 2 -> 3
                | 3 -> 4
            ",
            3.into(),
        ).unwrap();
    }

    #[test]
    fn test_match_int_with_equal_int_fail_matches_return_error2() {
        let err = end_to_end_test(
            "
            func 999
            . func = 
                | 1 -> 2
                | 2 -> 3
                | 3 -> 4
            ",
            2.into(),
        );

        assert_eq!(
            err, 
            Err((EvalError::MatchCaseNotFound(Node::Int { data: 999 }, vec![1.into(), 2.into(), 3.into()]), 18))
        );
    }

    #[test]
    fn test_eval_compose() {
        end_to_end_test(
            "
            ((a -> a + 1) >> (b -> b * 2)) 3
            ",
            8.into(),
        ).unwrap();
    }

    #[test]
    fn test_eval_compose_does_not_expose_internal_bbb() {
        let err = end_to_end_test(
            "
            f 3 . f = ((y -> bbb) >> (z -> aaa))
            ",
            8.into(),
        );

        assert_eq!(
            err, 
            Err((EvalError::VariableNotFoundInScope("bbb".to_string()), 30))
        );
        
    }

    #[test]
    fn test_eval_compose_does_not_expose_internal_aaa() {
        let err = end_to_end_test(
            "
            f 3 . f = ((y -> y) >> (z -> aaa))
            ",
            8.into(),
        );

        assert_eq!(
            err, 
            Err((EvalError::VariableNotFoundInScope("aaa".to_string()), 42))
        );
        
    }

    #[test]
    fn test_eval_compose_into_function() {
        end_to_end_test(
            "
            f 3 . f = ((y -> y) >> (z -> z * 3))
            ",
            9.into(),
        ).unwrap();
       
    }

    #[test]
    fn test_double_compose() {
        end_to_end_test(
            "
            ((a -> a + 1) >> (x -> x) >> (b -> b * 2)) 3
            ",
            8.into(),
        ).unwrap();
       
    }

    #[test]
    fn test_reverse_compose() {
        end_to_end_test(
            "
            ((a -> a + 1) << (b -> b * 2)) 3
            ",
            7.into(),
        ).unwrap();
       
    }

    #[test]
    fn test_bytes_return_bytes() {
        end_to_end_test(
            "
            ~~QUJD
            ",
            EvalResult::Bytes(vec![b'A', b'B', b'C']),
        ).unwrap();
       
    }

    #[test]
    fn test_bytes_base85_return_bytes() {
        end_to_end_test(
            "
            ~~85'K|(_
            ",
            EvalResult::Bytes(vec![b'A', b'B', b'C'])
        ).unwrap();
       
    }

    #[test]
    fn test_bytes_base64_return_bytes() {
        end_to_end_test(
            "
            ~~64'QUJD
            ",
            EvalResult::Bytes(vec![b'A', b'B', b'C'])
        ).unwrap();
       
    }

    #[test]
    fn test_bytes_base32_return_bytes() {
        end_to_end_test(
            "
            ~~32'IFBEG===
            ",
            EvalResult::Bytes(vec![b'A', b'B', b'C'])
        ).unwrap();
       
    }

    #[test]
    fn test_bytes_base16_return_bytes() {
        end_to_end_test(
            "
            ~~16'414243
            ",
            EvalResult::Bytes(vec![b'A', b'B', b'C'])
        ).unwrap();
    }

    #[test]
    fn test_int_add_returns_int() {
        end_to_end_test(
            "
            1 + 2
            ",
            3.into()
        ).unwrap();
    }

    #[test]
    fn test_int_sub_returns_int() {
        end_to_end_test(
            "
            1 - 2
            ",
            (-1).into()
        ).unwrap();
    }

    #[test]
    fn test_string_concat_returns_string() {
        end_to_end_test(
            "
            \"abc\" ++ \"def\"
            ",
            "abcdef".into()
        ).unwrap();
    }

    #[test]
    fn test_list_cons_nested_returns_list() {
        end_to_end_test(
            "
            1 >+ 2 >+ [3,4]
            ",
            vec![1.into(), 2.into(), 3.into(), 4.into()].into()
        ).unwrap();
    }

    #[test]
    fn test_list_append_returns_list() {
        end_to_end_test(
            "
            [1,2] +< 3
            ",
            vec![1.into(), 2.into(), 3.into()].into()
        ).unwrap();
    }

    #[test]
    fn test_list_append_nested_returns_list() {
        end_to_end_test(
            "
            [1,2] +< 3 +< 4
            ",
            vec![1.into(), 2.into(), 3.into(), 4.into()].into()
        ).unwrap();
    }

    #[test]
    fn test_where() {
        end_to_end_test(
            "
            a + 2
            . a = 1
            ",
            3.into()
        ).unwrap();
    }

    #[test]
    fn test_nested_where() {
        end_to_end_test(
            "
            a + b
            . a = 1
            . b = 4
            ",
            5.into()
        ).unwrap();
    }

    #[test]
    fn test_assert_with_truthy_cond_returns_value() {
        end_to_end_test(
            "
            a + 1 ? a == 1
            . a = 1
            ",
            2.into()
        ).unwrap();
    }

    #[test]
    fn test_assert_with_falsey_cond_returns_assertion() {
        let err = end_to_end_test(
            "
            a + 1 ? a == 2
            . a = 1
            ",
            2.into()
        );

        assert_eq!(err, Err((EvalError::FalseConditionFound, 23)));
    }

    #[test]
    fn test_nested_assert() {
        end_to_end_test(
            "
            a + b ? a == 1 ? b == 2
            . a = 1
            . b = 2
            ",
            3.into()
        ).unwrap();
    }

    #[test]
    fn test_bindings_behave_like_letstar() {
        end_to_end_test(
            "
            b
            . a = 1
            . b = a
            ",
            1.into()
        ).unwrap();
    }

    #[test]
    fn test_function_application_two_args() {
        end_to_end_test(
            "
            (a -> b -> a + b) 8 1
            ",
            9.into()
        ).unwrap();
    }

    #[test]
    fn test_function_create_list_correct_order() {
        end_to_end_test(
            "
            (a -> b -> [a, b]) 3 2
            ",
            vec![3.into(), 2.into()].into()
        ).unwrap();
    }

    #[test]
    fn test_access_record() {
        end_to_end_test(
            "
            rec@b
            . badvalue =  { a = 1, b = \"nope\" }
            . rec =  { a = 1 + 4, b = \"x\" }
            ",
            "x".into()
        ).unwrap();
    }

    #[test]
    fn test_access_list() {
        end_to_end_test(
            "
            xs@1
            . xs =  [0, 1, 2, 3, 4]
            ",
            1.into()
        ).unwrap();
    }

    #[test]
    fn test_access_list_expr() {
        end_to_end_test(
            "
            xs@(1+1)
            . xs =  [0, 1, 2, 3, 4]
            ",
            2.into()
        ).unwrap();
    }

    #[test]
    fn test_access_list_var() {
        end_to_end_test(
            "
            xs@y
            . xs = [0, 1, 2, 3, 4]
            . y  = 4
            ",
            4.into()
        ).unwrap();
    }

    #[test]
    fn test_functions_eval_arguments() {
        end_to_end_test(
            "
            (x -> x) c
            . c = 123
            ",
            123.into()
        ).unwrap();
    }

    #[test]
    fn test_compose() {
        end_to_end_test(
            "((a -> a + 1) >> (b -> b * 2)) 3",
            8.into()
        ).unwrap();
    }

    #[test]
    fn test_compose_does_not_expose_internal_x() {
        let err = end_to_end_test(
            "
            f 3
            . f = ((y -> x) >> (z -> x))
            ",
            8.into()
        );

        assert_eq!(err, Err((EvalError::VariableNotFoundInScope("x".to_string()), 42)));
    }

    #[test]
    fn test_double_compose_end_to_end() {
        end_to_end_test(
            "((a -> a + 1) >> (x -> x) >> (b -> b * 2)) 3",
            8.into()
        ).unwrap();
    }

    #[test]
    fn test_reverse_compose_end_to_end() {
        end_to_end_test(
            "((a -> a + 1) << (b -> b * 2)) 3",
            7.into()
        ).unwrap();
    }

    #[test]
    fn test_simple_int_match() {
        end_to_end_test(
            "
            inc 2
            . inc =
              | 1 -> 2
              | 2 -> 3
            ",
            3.into()
        ).unwrap();
    }

    #[test]
    fn test_match_var_binds_var() {
        end_to_end_test(
            "
            id 3
            . id =
              | x -> x
            ",
            3.into()
        ).unwrap();
    }

    #[test]
    fn test_match_var_binds_var_with_expr() {
        end_to_end_test(
            "
            id 3
            . id =
              | x -> x * 4 + 1
            ",
            13.into()
        ).unwrap();
    }

    #[test]
    fn test_match_var_binds_first_arm() {
        end_to_end_test(
            "
            id 3
            . id =
              | x -> x
              | y -> y * 2
            ",
            3.into()
        ).unwrap();
    }

    #[test]
    fn test_match_function_can_close_over_variables() {
        end_to_end_test(
            "
            f 2 10
            . f = a -> 
              | b -> a + b
            ",
            12.into()
        ).unwrap();
    }

    #[test]
    fn test_match_record_binds_var() {
        end_to_end_test(
            "get_x rec 
            . rec = { x = 5 } 
            . get_x =  
              | { x = x } -> x",
            5.into()
        ).unwrap();
    }
}


