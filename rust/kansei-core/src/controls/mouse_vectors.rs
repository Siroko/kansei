use crate::math::Vec2;

/// Mouse position/direction tracker (mirrors TS MouseVectors).
pub struct MouseVectors {
    pub position: Vec2,
    pub direction: Vec2,
    pub strength: f32,
    prev_position: Vec2,
}

impl MouseVectors {
    pub fn new() -> Self {
        Self {
            position: Vec2::ZERO,
            direction: Vec2::ZERO,
            strength: 0.0,
            prev_position: Vec2::ZERO,
        }
    }

    pub fn set_position(&mut self, x: f32, y: f32) {
        self.prev_position = self.position;
        self.position = Vec2::new(x, y);
    }

    pub fn update(&mut self, _dt: f32) {
        let dx = self.position.x - self.prev_position.x;
        let dy = self.position.y - self.prev_position.y;
        self.direction = Vec2::new(dx, dy);
        self.strength = self.direction.length().min(1.0);
    }
}

impl Default for MouseVectors {
    fn default() -> Self { Self::new() }
}
