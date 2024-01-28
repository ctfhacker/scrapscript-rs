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
    OpWhere,
    OpAssign,
    OpMatchCase,
    OpFunction,
    OpMul,
    OpDiv,
    OpExp,
    OpAdd,
    OpSub,
    OpEqual,
    OpMod,
    OpFloorDiv,
    OpNotEqua,
    OpLess,
    OpGreater,
    OpLessEqual,
    OpGreaterEqual,
    OpBoolAnd,
    OpBoolOr,
    OpStrConcat,
    OpListCons,
    OpListAppend,
    OpRightEval,
    OpHasType,
    OpPipe,
    OpReversePipe,
    OpSpread,
    OpAssert,
    EndOfFile,
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
}

/*
PS = {
    "::": lp(2000),
    "@": rp(1001),
    "": rp(1000),
    ">>": lp(14),
    "<<": lp(14),
    "^": rp(13),
    "*": lp(12),
    "/": lp(12),
    "//": lp(12),
    "%": lp(12),
    "+": lp(11),
    "-": lp(11),
    ">*": rp(10),
    "++": rp(10),
    ">+": lp(10),
    "+<": rp(10),
    "==": np(9),
    "/=": np(9),
    "<": np(9),
    ">": np(9),
    "<=": np(9),
    ">=": np(9),
    "&&": rp(8),
    "||": rp(7),
    "|>": rp(6),
    "<|": lp(6),
    "->": lp(5),
    "|": rp(4.5),
    ":": lp(4.5),
    "=": rp(4),
    "!": lp(3),
    ".": rp(3),
    "?": rp(3),
    ",": xp(1),
    # TODO: Fix precedence for spread
    "...": xp(0),
}
*/

/// Convert the input program into a list of tokens
pub fn tokenize(input: &str) -> Result<Vec<Token>, (TokenError, usize)> {
    let mut result = Vec::new();
    let input: &str = input.into();
    let input = input.as_bytes();

    let mut index = 0;
    let mut in_token = false;

    let mut pos = index;

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
                            pos: index as u32,
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
                            pos: index as u32,
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
                            pos: index as u32,
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
                        pos: index as u32,
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
                continue_while!(b'!'..=b'u');
            }
            [b'~', b'~', b'3', b'2', b'\'', ..] => {
                pos += 5;
                index += 5;
                set_token!(Base32, 5);
                continue_while!(b'A'..=b'Z' | b'2'..=b'7' | b'=');
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
            [b'=', b'=', ..] => {
                set_token!(OpEqual, 2);
            }
            [b'/', b'/', ..] => {
                set_token!(OpFloorDiv, 2);
            }
            [b'/', b'=', ..] => {
                set_token!(OpNotEqua, 2);
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
            [b':', ..] => {
                set_token!(OpHasType, 1);
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
                index += 1;
            }
        }
    }

    Ok(result)
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
            ("/=", OpNotEqua),
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
    fn test_tokenize_base64() {
        let tokens = tokenize("~~QUJD");

        assert_eq!(
            tokens,
            Ok(vec![Token::new(Base64, 2), Token::new(EndOfFile, 6)])
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
}
