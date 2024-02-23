//! Implements ANSI foreground colors for printable types

use core::marker::PhantomData;

/// Wrapper struct for generically selecting a color over something to be printed
pub struct Styled<'a, T: Sized, C> {
    /// Original data to stylize
    original: &'a T,

    /// Color of the text
    color: PhantomData<C>,
}

/// Trait which provides the ANSI color string
pub trait Color {
    /// The ANSI color string for this color
    const ANSI: &'static str;
}

/// Creates structs for each ANSI color and implemented the [`Color`] trait for them
macro_rules! create_color {
    ($color:ident, $bold:expr, $num:expr) => {
        pub struct $color;

        impl Color for $color {
            const ANSI: &'static str =
                concat!("\x1b[", stringify!($bold), ";", stringify!($num), "m");
        }
    };
}

create_color!(Black, 0, 30);
create_color!(Red, 0, 31);
create_color!(Green, 0, 32);
create_color!(Yellow, 0, 33);
create_color!(Blue, 0, 34);
create_color!(Magenta, 0, 35);
create_color!(Cyan, 0, 36);
create_color!(White, 0, 37);
create_color!(Normal, 0, 39);
create_color!(BrightBlack, 0, 90);
create_color!(BrightRed, 0, 91);
create_color!(BrightGreen, 0, 92);
create_color!(BrightYellow, 0, 93);
create_color!(BrightBlue, 0, 94);
create_color!(BrightMagenta, 0, 95);
create_color!(BrightCyan, 0, 96);
create_color!(BrightWhite, 0, 97);
create_color!(BoldBlack, 1, 30);
create_color!(BoldRed, 1, 31);
create_color!(BoldGreen, 1, 32);
create_color!(BoldYellow, 1, 33);
create_color!(BoldBlue, 1, 34);
create_color!(BoldMagenta, 1, 35);
create_color!(BoldCyan, 1, 36);
create_color!(BoldWhite, 1, 37);
create_color!(BoldNormal, 1, 39);
create_color!(BoldBrightBlack, 1, 90);
create_color!(BoldBrightRed, 1, 91);
create_color!(BoldBrightGreen, 1, 92);
create_color!(BoldBrightYellow, 1, 93);
create_color!(BoldBrightBlue, 1, 94);
create_color!(BoldBrightMagenta, 1, 95);
create_color!(BoldBrightCyan, 1, 96);
create_color!(BoldBrightWhite, 1, 97);

/// Implements the various `core::fmt` traits
macro_rules! impl_formats {
    ($ty:path) => {
        impl<'a, T: $ty, C: Color> $ty for Styled<'a, T, C> {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                let _ = f.write_str(C::ANSI);
                let _ = <T as $ty>::fmt(&self.original, f);
                f.write_str(Normal::ANSI)
            }
        }
    };
}

/// Implements each function that is available for adding color
macro_rules! trait_func {
    ($color:ident, $ty:ident) => {
        #[allow(dead_code)]
        fn $color(&self) -> Styled<Self, $ty>
        where
            Self: Sized,
        {
            Styled {
                original: self,
                color: PhantomData,
            }
        }
    };
}

/// Provides wrapper functions to apply foreground colors
pub trait Colorized {
    trait_func!(black, Black);
    trait_func!(red, Red);
    trait_func!(green, Green);
    trait_func!(yellow, Yellow);
    trait_func!(blue, Blue);
    trait_func!(magenta, Magenta);
    trait_func!(cyan, Cyan);
    trait_func!(white, White);
    trait_func!(normal, Normal);
    trait_func!(default, Normal);
    trait_func!(bright_black, BrightBlack);
    trait_func!(bright_red, BrightRed);
    trait_func!(bright_green, BrightGreen);
    trait_func!(bright_yellow, BrightYellow);
    trait_func!(bright_blue, BrightBlue);
    trait_func!(bright_magenta, BrightMagenta);
    trait_func!(bright_cyan, BrightCyan);
    trait_func!(bright_white, BrightWhite);
    trait_func!(light_black, BrightBlack);
    trait_func!(light_red, BrightRed);
    trait_func!(light_green, BrightGreen);
    trait_func!(light_yellow, BrightYellow);
    trait_func!(light_blue, BrightBlue);
    trait_func!(light_magenta, BrightMagenta);
    trait_func!(light_cyan, BrightCyan);
    trait_func!(light_white, BrightWhite);
    trait_func!(bold_black, BoldBlack);
    trait_func!(bold_red, BoldRed);
    trait_func!(bold_green, BoldGreen);
    trait_func!(bold_yellow, BoldYellow);
    trait_func!(bold_blue, BoldBlue);
    trait_func!(bold_magenta, BoldMagenta);
    trait_func!(bold_cyan, BoldCyan);
    trait_func!(bold_white, BoldWhite);
    trait_func!(bold_normal, BoldNormal);
    trait_func!(bold_default, BoldNormal);
    trait_func!(bold_bright_black, BoldBrightBlack);
    trait_func!(bold_bright_red, BoldBrightRed);
    trait_func!(bold_bright_green, BoldBrightGreen);
    trait_func!(bold_bright_yellow, BoldBrightYellow);
    trait_func!(bold_bright_blue, BoldBrightBlue);
    trait_func!(bold_bright_magenta, BoldBrightMagenta);
    trait_func!(bold_bright_cyan, BoldBrightCyan);
    trait_func!(bold_bright_white, BoldBrightWhite);
    trait_func!(bold_light_black, BoldBrightBlack);
    trait_func!(bold_light_red, BoldBrightRed);
    trait_func!(bold_light_green, BoldBrightGreen);
    trait_func!(bold_light_yellow, BoldBrightYellow);
    trait_func!(bold_light_blue, BoldBrightBlue);
    trait_func!(bold_light_magenta, BoldBrightMagenta);
    trait_func!(bold_light_cyan, BoldBrightCyan);
    trait_func!(bold_light_white, BoldBrightWhite);
}

impl_formats!(core::fmt::Debug);
impl_formats!(core::fmt::Display);
impl_formats!(core::fmt::LowerHex);

// Magic impl which gives any trait that implements Display color functions
impl<T: core::fmt::Display> Colorized for T {}
