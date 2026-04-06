use crate::math::Vec2;

/// Mouse position/direction tracker (mirrors TS MouseVectors).
pub struct MouseVectors {
    pub speed: f32,
    pub position: Vec2,
    pub direction: Vec2,
    pub strength: f32,
    end: Vec2,
}

impl MouseVectors {
    pub fn new() -> Self {
        Self {
            speed: 8.0,
            position: Vec2::ZERO,
            direction: Vec2::ZERO,
            strength: 0.0,
            end: Vec2::ZERO,
        }
    }

    /// Set an immediate position (and target), avoiding interpolation jumps.
    pub fn set_position(&mut self, x: f32, y: f32) {
        self.position = Vec2::new(x, y);
        self.end = self.position;
        self.direction = Vec2::ZERO;
        self.strength = 0.0;
    }

    /// Set the interpolation target in NDC space [-1, 1].
    pub fn set_target(&mut self, x: f32, y: f32) {
        self.end = Vec2::new(x, y);
    }

    /// Set the interpolation target from pixel coordinates.
    pub fn set_target_from_screen(&mut self, x: f32, y: f32, width: f32, height: f32) {
        let w = width.max(1.0);
        let h = height.max(1.0);
        self.set_target((x / w - 0.5) * 2.0, (y / h - 0.5) * 2.0);
    }

    /// Set immediate position from pixel coordinates.
    pub fn set_position_from_screen(&mut self, x: f32, y: f32, width: f32, height: f32) {
        let w = width.max(1.0);
        let h = height.max(1.0);
        self.set_position((x / w - 0.5) * 2.0, (y / h - 0.5) * 2.0);
    }

    pub fn update(&mut self, dt: f32) {
        let prev = self.position;
        let t = self.speed * dt.max(0.0);
        self.position = Vec2::new(
            self.position.x + (self.end.x - self.position.x) * t,
            self.position.y + (self.end.y - self.position.y) * t,
        );
        self.direction = Vec2::new(prev.x - self.position.x, prev.y - self.position.y);
        self.strength = self.direction.length();
    }
}

impl Default for MouseVectors {
    fn default() -> Self { Self::new() }
}
