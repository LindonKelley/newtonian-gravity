use std::marker::PhantomData;
use std::mem;
use std::ops::{Deref, DerefMut};
use image::ImageBuffer;
use num_traits::ToPrimitive;

pub struct CPURenderer<C, FH: FrameHandler<C>, P, R: Rasterizer<C, P>> {
    frame_handler: FH,
    // todo current implementation means I can just remove the object here, since there's no
    //  function with Self required in Rasterizer, could also just change over to a function instead
    rasterizer: R,
    __phantom: PhantomData<(C, P)>
}

impl <C, FH: FrameHandler<C>, P, R: Rasterizer<C, P>> CPURenderer<C, FH, P, R> {
    pub fn new(frame_handler: FH, rasterizer: R) -> Self {
        Self {
            frame_handler,
            rasterizer,
            __phantom: PhantomData
        }
    }
}

pub trait FrameHandler<Canvas> {
    fn produce(&self) -> Canvas;

    fn consume(&self, canvas: Canvas);
}

pub trait Rasterizer<Canvas, Paint> {
    fn draw_filled_circle(canvas: &mut Canvas, cx: f32, cy: f32, r: f32, color: Paint);
}

pub trait FixedSizeCanvas {
    fn width(&self) -> u32;

    fn height(&self) -> u32;
}

pub trait HorizontalLineCanvas<Paint>: FixedSizeCanvas {
    unsafe fn draw_horizontal_line_unchecked(&mut self, x0: u32, x1: u32, y: u32, color: Paint);
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

impl <Pixel: image::Pixel, Container: Deref<Target = [Pixel::Subpixel]> + DerefMut> From<ImageBuffer<Pixel, Container>> for HorizontalLineImage<Pixel, Container> {
    fn from(image: ImageBuffer<Pixel, Container>) -> Self {
        Self {
            width: image.width(),
            height: image.height(),
            data: image.into_raw(),
            __phantom: PhantomData
        }
    }
}

impl <Pixel: image::Pixel, Container: Deref<Target = [Pixel::Subpixel]> + DerefMut> From<HorizontalLineImage<Pixel, Container>> for ImageBuffer<Pixel, Container> {
    fn from(image: HorizontalLineImage<Pixel, Container>) -> Self {
        // SAFETY: limited construction avenues for HorizontalLineImage prevent this from being invalid
        unsafe {
            Self::from_raw(image.width, image.height, image.data).unwrap_unchecked()
        }
    }
}

pub struct FastIntegerRasterizer;

impl <Paint: Copy, Canvas: HorizontalLineCanvas<Paint>> Rasterizer<Canvas, Paint> for FastIntegerRasterizer {
    fn draw_filled_circle(canvas: &mut Canvas, cx: f32, cy: f32, r: f32, color: Paint) {
        Self::draw_filled_circle_internal(canvas, cx, cy, r, color);
    }
}

impl FastIntegerRasterizer {
    // this implementation doesn't draw circles beyond i32::MAX in order to save on a bit of speed
    // (and my sanity), and because rendering an image that large seems a bit extreme
    fn draw_filled_circle_internal<Paint: Copy, Canvas: HorizontalLineCanvas<Paint>>(canvas: &mut Canvas, cx: f32, cy: f32, r: f32, color: Paint) -> Option<()> {
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
                    Self::draw_horizontal_line(canvas, sub_x0_x, add_x0_x, y0.checked_sub(y), color);
                    Self::draw_horizontal_line(canvas, sub_x0_x, add_x0_x, y0.checked_add(y), color);
                }
            }
            let sub_x0_y = x0.saturating_sub(y).try_into().unwrap_or(0);
            if sub_x0_y < canvas.width() {
                let add_x0_y = u32::min(x0.saturating_add(y).try_into().unwrap_or(0) + 1, canvas.width());
                unsafe {
                    Self::draw_horizontal_line(canvas, sub_x0_y, add_x0_y, y0.checked_sub(x), color);
                    Self::draw_horizontal_line(canvas, sub_x0_y, add_x0_y, y0.checked_add(x), color);
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

    unsafe fn draw_horizontal_line<Paint: Copy, Canvas: HorizontalLineCanvas<Paint>>(canvas: &mut Canvas, x0: u32, x1: u32, opt_signed_y: Option<i32>, color: Paint) {
        if let Some(signed_y) = opt_signed_y {
            if signed_y >= 0 {
                let y = signed_y as u32;
                if y < canvas.height() {
                    canvas.draw_horizontal_line_unchecked(x0, x1, y, color);
                }
            }
        }
    }
}
