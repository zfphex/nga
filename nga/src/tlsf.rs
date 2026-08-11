pub const NIL: u32 = u32::MAX;
pub const MIN: u64 = 256;

const FL_SHIFT: usize = 8;
const FL: usize = 32;
const SL_BITS: usize = 5;
const SL: usize = 1 << SL_BITS;

#[derive(Clone, Copy, Debug)]
pub struct Alloc {
    pub offset: u64,
    pub size: u64,
    pub block: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct Block {
    pub offset: u64,
    pub size: u64,
    pub prev: u32,
    pub next: u32,
    pub free_prev: u32,
    pub free_next: u32,
    pub free: bool,
}

pub struct Tlsf {
    pub blocks: Vec<Block>,
    pub spare: Vec<u32>,
    pub heads: [[u32; SL]; FL],
    pub sl_bitmap: [u32; FL],
    pub fl_bitmap: u32,
    pub size: u64,
}

impl Tlsf {
    pub fn new(size: u64) -> Self {
        assert!(size >= MIN && size < MIN << FL);
        let mut tlsf = Self {
            blocks: Vec::new(),
            spare: Vec::new(),
            heads: [[NIL; SL]; FL],
            sl_bitmap: [0; FL],
            fl_bitmap: 0,
            size,
        };
        let root = tlsf.block(0, size & !(MIN - 1));
        tlsf.insert(root);
        tlsf
    }

    fn mapping(size: u64) -> (usize, usize) {
        let fl = 63 - size.leading_zeros() as usize;
        let sl = ((size >> (fl - SL_BITS)) & (SL as u64 - 1)) as usize;
        (fl - FL_SHIFT, sl)
    }

    fn block(&mut self, offset: u64, size: u64) -> u32 {
        let block = Block {
            offset,
            size,
            prev: NIL,
            next: NIL,
            free_prev: NIL,
            free_next: NIL,
            free: false,
        };
        match self.spare.pop() {
            Some(index) => {
                self.blocks[index as usize] = block;
                index
            }
            None => {
                self.blocks.push(block);
                self.blocks.len() as u32 - 1
            }
        }
    }

    fn insert(&mut self, index: u32) {
        let (fl, sl) = Self::mapping(self.blocks[index as usize].size);
        let head = self.heads[fl][sl];
        self.blocks[index as usize].free = true;
        self.blocks[index as usize].free_prev = NIL;
        self.blocks[index as usize].free_next = head;
        if head != NIL {
            self.blocks[head as usize].free_prev = index;
        }
        self.heads[fl][sl] = index;
        self.fl_bitmap |= 1 << fl;
        self.sl_bitmap[fl] |= 1 << sl;
    }

    fn remove(&mut self, index: u32) {
        let block = self.blocks[index as usize];
        let (fl, sl) = Self::mapping(block.size);
        if block.free_prev != NIL {
            self.blocks[block.free_prev as usize].free_next = block.free_next;
        } else {
            self.heads[fl][sl] = block.free_next;
        }
        if block.free_next != NIL {
            self.blocks[block.free_next as usize].free_prev = block.free_prev;
        }
        if self.heads[fl][sl] == NIL {
            self.sl_bitmap[fl] &= !(1 << sl);
            if self.sl_bitmap[fl] == 0 {
                self.fl_bitmap &= !(1 << fl);
            }
        }
        self.blocks[index as usize].free = false;
    }

    pub fn alloc(&mut self, size: u64, align: u64) -> Option<Alloc> {
        let align = align.max(MIN).next_power_of_two();
        let size = (size.max(1) + MIN - 1) & !(MIN - 1);
        let want = size + if align > MIN { align } else { 0 };
        let raw = 63 - want.leading_zeros() as usize;
        let (mut fl, mut sl) = Self::mapping(want + (1 << (raw - SL_BITS)) - 1);

        let mut map = self.sl_bitmap[fl] & (u32::MAX << sl);
        if map == 0 {
            let up = self.fl_bitmap & u32::MAX.checked_shl(fl as u32 + 1).unwrap_or(0);
            if up == 0 {
                return None;
            }
            fl = up.trailing_zeros() as usize;
            map = self.sl_bitmap[fl];
        }
        sl = map.trailing_zeros() as usize;

        let index = self.heads[fl][sl];
        self.remove(index);
        let (offset, total) = (
            self.blocks[index as usize].offset,
            self.blocks[index as usize].size,
        );
        let aligned = (offset + align - 1) & !(align - 1);

        if aligned > offset {
            let pad = self.block(offset, aligned - offset);
            let prev = self.blocks[index as usize].prev;
            self.blocks[pad as usize].prev = prev;
            self.blocks[pad as usize].next = index;
            if prev != NIL {
                self.blocks[prev as usize].next = pad;
            }
            self.blocks[index as usize].prev = pad;
            self.blocks[index as usize].offset = aligned;
            self.blocks[index as usize].size = total - (aligned - offset);
            self.insert(pad);
        }

        let rest = self.blocks[index as usize].size - size;
        if rest >= MIN {
            let tail = self.block(aligned + size, rest);
            let next = self.blocks[index as usize].next;
            self.blocks[tail as usize].prev = index;
            self.blocks[tail as usize].next = next;
            if next != NIL {
                self.blocks[next as usize].prev = tail;
            }
            self.blocks[index as usize].next = tail;
            self.blocks[index as usize].size = size;
            self.insert(tail);
        }

        Some(Alloc {
            offset: aligned,
            size: self.blocks[index as usize].size,
            block: index,
        })
    }

    pub fn free(&mut self, alloc: Alloc) {
        let mut index = alloc.block;
        let next = self.blocks[index as usize].next;
        if next != NIL && self.blocks[next as usize].free {
            self.remove(next);
            let after = self.blocks[next as usize].next;
            self.blocks[index as usize].size += self.blocks[next as usize].size;
            self.blocks[index as usize].next = after;
            if after != NIL {
                self.blocks[after as usize].prev = index;
            }
            self.spare.push(next);
        }
        let prev = self.blocks[index as usize].prev;
        if prev != NIL && self.blocks[prev as usize].free {
            self.remove(prev);
            let after = self.blocks[index as usize].next;
            self.blocks[prev as usize].size += self.blocks[index as usize].size;
            self.blocks[prev as usize].next = after;
            if after != NIL {
                self.blocks[after as usize].prev = prev;
            }
            self.spare.push(index);
            index = prev;
        }
        self.insert(index);
    }
}
