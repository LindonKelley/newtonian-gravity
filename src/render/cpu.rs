use crate::render::cpu;
use std::marker::PhantomData;
use std::mem;
use std::ops::{Deref, DerefMut};
use num_traits::ToPrimitive;

pub struct CPURenderer<
    Canvas,
    Paint,
    PaintScalar: cpu::PaintScalar<Paint>,
    FrameHandler: cpu::FrameHandler<Canvas>,
    Rasterizer: cpu::Rasterizer<Canvas, Paint, PaintScalar>
> {
    frame_handler: FrameHandler,
    __phantom: PhantomData<(Canvas, Paint, PaintScalar, Rasterizer)>
}

impl <
    Canvas,
    Paint,
    PaintScalar: cpu::PaintScalar<Paint>,
    FrameHandler: cpu::FrameHandler<Canvas>,
    Rasterizer: cpu::Rasterizer<Canvas, Paint, PaintScalar>
> CPURenderer<Canvas, Paint, PaintScalar, FrameHandler, Rasterizer> {
    pub fn new(frame_handler: FrameHandler) -> Self {
        Self {
            frame_handler,
            __phantom: PhantomData
        }
    }
}

pub trait FrameHandler<Canvas> {
    fn produce(&self) -> Canvas;

    fn consume(&self, canvas: Canvas);
}

pub trait PaintScalar<Paint> {
    fn scale(paint: &Paint, scale: f32, clamp: Option<fn(f32) -> f32>) -> Paint;
}

pub struct GrayscaleRgbScalar;

impl PaintScalar<image::Rgb<u8>> for GrayscaleRgbScalar {
    fn scale(paint: &image::Rgb<u8>, scale: f32, _: Option<fn(f32) -> f32>) -> image::Rgb<u8> {
        // converting f32 to u8 through `as` is a clamping operation, so `clamp` can be ignored
        let c = (paint.0[0] as f32 * scale) as u8;
        [c; 3].into()
    }
}

pub trait Rasterizer<Canvas, Paint, Scalar: PaintScalar<Paint>> {
    // r should not be negative
    fn draw_filled_circle(canvas: &mut Canvas, cx: f32, cy: f32, r: f32, paint: Paint);
}

pub trait FixedSizeCanvas {
    fn width(&self) -> u32;

    fn height(&self) -> u32;
}

pub trait HorizontalLineCanvas<Paint>: FixedSizeCanvas {
    unsafe fn draw_pixel_unchecked(&mut self, x: u32, y: u32, paint: Paint);

    unsafe fn draw_horizontal_line_unchecked(&mut self, x0: u32, x1: u32, y: u32, paint: Paint);
}

/// `HorizontalLineImage` represents an image, supports fast horizontal line drawing, and is
/// convertible from and to image::ImageBuffer
pub struct HorizontalLineImage<Pixel: image::Pixel, Container: Deref<Target = [Pixel::Subpixel]> + DerefMut> {
    width: u32,
    height: u32,
    data: Container,
    __phantom: PhantomData<Pixel>
}

impl <Pixel: image::Pixel, Container: Deref<Target = [Pixel::Subpixel]> + DerefMut> HorizontalLineImage<Pixel, Container> {
    #[inline(always)]
    fn to_data_index(&self, x: u32, y: u32) -> usize {
        (y as usize * self.width as usize + x as usize) * mem::size_of::<Pixel>()
    }
}

impl<Pixel: image::Pixel, Container: Deref<Target=[Pixel::Subpixel]> + DerefMut> FixedSizeCanvas for HorizontalLineImage<Pixel, Container> {
    #[inline(always)]
    fn width(&self) -> u32 {
        self.width
    }

    #[inline(always)]
    fn height(&self) -> u32 {
        self.height
    }
}

impl <Pixel: image::Pixel, Container: Deref<Target = [Pixel::Subpixel]> + DerefMut> HorizontalLineCanvas<Pixel> for HorizontalLineImage<Pixel, Container> {
    unsafe fn draw_pixel_unchecked(&mut self, x: u32, y: u32, color: Pixel) {
        let index = self.to_data_index(x, y);
        let ptr = self.data.get_unchecked_mut(index) as *mut _ as *mut Pixel;
        *ptr = color;
    }

    unsafe fn draw_horizontal_line_unchecked(&mut self, x0: u32, x1: u32, y: u32, color: Pixel) {
        let start = self.to_data_index(x0, y);
        let mut addr = self.data.get_unchecked_mut(start) as *mut _ as usize;
        let end = self.to_data_index(x1, y);
        let end_addr = self.data.get_unchecked(end) as *const _ as usize;
        while addr < end_addr {
            // SAFETY: image::Pixel is Copy
            *(addr as *mut Pixel) = color;
            addr += mem::size_of::<Pixel>();
        }
    }
}

impl <Pixel: image::Pixel, Container: Deref<Target = [Pixel::Subpixel]> + DerefMut> From<image::ImageBuffer<Pixel, Container>> for HorizontalLineImage<Pixel, Container> {
    fn from(image: image::ImageBuffer<Pixel, Container>) -> Self {
        Self {
            width: image.width(),
            height: image.height(),
            data: image.into_raw(),
            __phantom: PhantomData
        }
    }
}

impl <Pixel: image::Pixel, Container: Deref<Target = [Pixel::Subpixel]> + DerefMut> From<HorizontalLineImage<Pixel, Container>> for image::ImageBuffer<Pixel, Container> {
    fn from(image: HorizontalLineImage<Pixel, Container>) -> Self {
        // SAFETY: limited construction avenues for HorizontalLineImage prevent this from being invalid
        unsafe {
            Self::from_raw(image.width, image.height, image.data).unwrap_unchecked()
        }
    }
}

pub struct FastIntegerRasterizer;

impl <Paint: Copy, Canvas: HorizontalLineCanvas<Paint>, Scalar: PaintScalar<Paint>> Rasterizer<Canvas, Paint, Scalar> for FastIntegerRasterizer {
    fn draw_filled_circle(canvas: &mut Canvas, cx: f32, cy: f32, r: f32, paint: Paint) {
        Self::draw_filled_circle_internal(canvas, cx, cy, r, paint);
    }
}

impl FastIntegerRasterizer {
    // this implementation doesn't draw circles beyond i32::MAX in order to save on a bit of speed
    // (and my sanity), and because rendering an image that large seems a bit extreme
    #[inline(always)]
    fn draw_filled_circle_internal<Paint: Copy, Canvas: HorizontalLineCanvas<Paint>>(canvas: &mut Canvas, cx: f32, cy: f32, r: f32, paint: Paint) -> Option<()> {
        let x0 = cx.to_i32()?;
        let y0 = cy.to_i32()?;
        let r = r.to_i32()?;
        let mut x = 0;
        let mut y = r;
        let mut p = 1 - r;
        while x <= y {
            let sub_x0_x = x0.saturating_sub(x).try_into().unwrap_or(0);
            if sub_x0_x < canvas.width() {
                let add_x0_x = u32::min(x0.saturating_add(x).try_into().unwrap_or(0) + 1, canvas.width());
                unsafe {
                    Self::draw_horizontal_line(canvas, sub_x0_x, add_x0_x, y0.checked_sub(y), paint);
                    Self::draw_horizontal_line(canvas, sub_x0_x, add_x0_x, y0.checked_add(y), paint);
                }
            }
            let sub_x0_y = x0.saturating_sub(y).try_into().unwrap_or(0);
            if sub_x0_y < canvas.width() {
                let add_x0_y = u32::min(x0.saturating_add(y).try_into().unwrap_or(0) + 1, canvas.width());
                unsafe {
                    Self::draw_horizontal_line(canvas, sub_x0_y, add_x0_y, y0.checked_sub(x), paint);
                    Self::draw_horizontal_line(canvas, sub_x0_y, add_x0_y, y0.checked_add(x), paint);
                }
            }

            x += 1;
            if p < 0 {
                p += 2 * x + 1;
            } else {
                y -= 1;
                p += 2 * (x - y) + 1;
            }
        }
        Some(())
    }

    unsafe fn draw_horizontal_line<Paint: Copy, Canvas: HorizontalLineCanvas<Paint>>(canvas: &mut Canvas, x0: u32, x1: u32, opt_signed_y: Option<i32>, paint: Paint) {
        if let Some(signed_y) = opt_signed_y {
            if signed_y >= 0 {
                let y = signed_y as u32;
                if y < canvas.height() {
                    canvas.draw_horizontal_line_unchecked(x0, x1, y, paint);
                }
            }
        }
    }
}

pub struct AreaIntersectionRasterizer;

impl <Paint: Copy, Canvas: HorizontalLineCanvas<Paint>, Scalar: PaintScalar<Paint>> Rasterizer<Canvas, Paint, Scalar> for AreaIntersectionRasterizer {
    fn draw_filled_circle(canvas: &mut Canvas, cx: f32, cy: f32, r: f32, paint: Paint) {
        let min_x = (cx - r) as u32;
        if min_x >= canvas.width() {
            return;
        }

        let min_y = (cy - r) as u32;
        if min_y >= canvas.height() {
            return;
        }

        let max_y = u32::min((cy + r + 1.0) as u32, canvas.height());

        if r <= 2.0 {
            let max_x = u32::min((cx + r + 1.0) as u32, canvas.width());

            for y in min_y..max_y {
                let y0 = y as f32;
                let y1 = y0 + 1.0;
                for x in min_x..max_x {
                    let x0 = x as f32;
                    let x1 = x0 + 1.0;
                    let a = area_intersection_circle_rectangle(x0, y0, x1, y1, cx, cy, r);
                    let scaled_paint = Scalar::scale(&paint, a, Some(|f| clamp(f, 0.0, 1.0)));
                    unsafe {
                        canvas.draw_pixel_unchecked(x, y, scaled_paint);
                    }
                }
            }
        } else {
            let r_sq = r * r;

            // Rust seems to inline this
            let mut draw = |r_x: f32, y, y0, y1| {
                let min_x = (cx - r_x) as u32;
                let max_x = u32::min((cx + r_x) as u32 + 1, canvas.width());

                for x in min_x..max_x {
                    let x0 = x as f32;
                    let x1 = x0 + 1.0;
                    let a = area_intersection_circle_rectangle(x0, y0, x1, y1, cx, cy, r);
                    let scaled_paint = Scalar::scale(&paint, a, Some(|f| f32::min(f, 1.0)));
                    unsafe {
                        canvas.draw_pixel_unchecked(x, y, scaled_paint);
                    }
                }
            };

            for y in min_y..cy as u32 {
                let y0 = y as f32;
                let y1 = y0 + 1.0;
                let r_x = f32::sqrt(r_sq - ((y1 - cy) * (y1 - cy)));
                draw(r_x, y, y0, y1);
            }
            for y in cy as u32..max_y {
                let y0 = y as f32;
                let y1 = y0 + 1.0;
                let r_x = f32::sqrt(r_sq - ((y0 - cy) * (y0 - cy)));
                draw(r_x, y, y0, y1);
            }
        }
    }
}

/// Intersectional area of a rectangle and a circle
///
/// The rectangle's left edge is at `x0`, right edge is at `x1`, bottom edge is at `y0`, and top edge is at `y1`
///
/// The circle is centered at (`cx`, `cy`) with a radius of `r`
///
/// the following must hold true for correct results
/// x0 <= x1
/// y0 <= y1
/// r >= 0.0
fn area_intersection_circle_rectangle(x0: f32, y0: f32, x1: f32, y1: f32, cx: f32, cy: f32, r: f32) -> f32 {
    area_intersection_fixed_circle_rectangle(x0 - cx, y0 - cy, x1 - cx, y1 - cy, r)
}

/// Intersectional area of a rectangle and a circle
///
/// The rectangle's left edge is at `x0`, right edge is at `x1`, bottom edge is at `y0`, and top edge is at `y1`
///
/// The circle is centered at (0.0, 0.0) with a radius of `r`
///
/// the following must hold true for correct results
/// x0 <= x1
/// y0 <= y1
/// r >= 0.0
#[inline]
fn area_intersection_fixed_circle_rectangle(x0: f32, y0: f32, x1: f32, y1: f32, r: f32) -> f32 {
    debug_assert!(x0 <= x1);
    debug_assert!(y0 <= y1);
    debug_assert!(r >= 0.0);
    if y0 < 0.0 {
        if y1 < 0.0 {
            area_intersection_fixed_circle_rectangle(x0, -y1, x1, -y0, r)
        } else {
            area_intersection_fixed_circle_rectangle(x0, 0.0, x1, -y0, r) + area_intersection_fixed_circle_rectangle(x0, 0.0, x1, y1, r)
        }
    } else {
        area_intersection_fixed_circle_tall_rectangle(x0, x1, y0, r) - area_intersection_fixed_circle_tall_rectangle(x0, x1, y1, r)
    }
}

/// Intersectional area of an infinitely tall rectangle and a circle
///
/// The rectangle's left edge is at `x0`, right edge is at `x1`, bottom edge is at `h`, and top edge is at `f32::inf`
///
/// The circle is centered at (0.0, 0.0) with a radius of `r`
fn area_intersection_fixed_circle_tall_rectangle(x0: f32, x1: f32, h: f32, r: f32) -> f32 {
    let s = if h < r {
        f32::sqrt(r * r - h * h)
    } else {
        0.0
    };
    g(clamp(x1, -s, s), h, r) - g(clamp(x0, -s, s), h, r)
}

#[inline(always)]
fn clamp(v: f32, min: f32, max: f32) -> f32 {
    debug_assert!(min <= max);
    f32::max(min, f32::min(v, max))
}

/// Indefinite integral of a circle segment
#[inline(always)]
fn g(x: f32, h: f32, r: f32) -> f32 {
    (f32::sqrt(1.0 - x * x / (r * r)) * x * r + r * r * f32::asin(x / r) - 2.0 * h * x) / 2.0
}
