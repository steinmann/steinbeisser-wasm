use std::fs;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

pub const BINARY_SAMPLE_EXTENSION: &str = "sbin";
const MAGIC: &[u8; 8] = b"SBSMP01\n";

#[derive(Clone, Debug)]
pub struct BinarySample {
    pub black_bits: u64,
    pub white_bits: u64,
    pub side_to_move_is_black: bool,
    pub ply: f32,
    pub no_progress_plies: f32,
    pub score: f32,
    pub clipped_score: f32,
    pub result: f32,
    pub result_bucket: i32,
    pub completed_depth: f32,
    pub nodes: u64,
    pub elapsed_ms: u64,
    pub caused_ejection: bool,
    pub occurrence_count: u32,
    pub sample_weight: f32,
}

impl BinarySample {
    pub fn key(&self) -> String {
        format!(
            "{:016x}:{:016x}:{}:{:08x}:{:08x}",
            self.black_bits,
            self.white_bits,
            if self.side_to_move_is_black { "b" } else { "w" },
            self.ply.to_bits(),
            self.no_progress_plies.to_bits(),
        )
    }
}

pub fn write_samples(path: &Path, samples: &[BinarySample]) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = path.with_file_name(format!(
        ".{}.tmp",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("samples.sbin")
    ));
    {
        let mut writer = BufWriter::new(fs::File::create(&tmp)?);
        write_header(&mut writer)?;
        for sample in samples {
            write_record(&mut writer, sample)?;
        }
        writer.flush()?;
    }
    fs::rename(tmp, path)?;
    Ok(())
}

pub fn copy_prefix(source: &Path, destination: &Path, limit: Option<usize>) -> io::Result<usize> {
    let samples = read_samples(source)?;
    let count = limit.map_or(samples.len(), |limit| limit.min(samples.len()));
    write_samples(destination, &samples[..count])?;
    Ok(count)
}

pub fn read_samples(path: &Path) -> io::Result<Vec<BinarySample>> {
    let mut reader = BufReader::new(fs::File::open(path)?);
    let mut magic = [0_u8; MAGIC.len()];
    reader.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{} has an unsupported sample binary header", path.display()),
        ));
    }
    let mut samples = Vec::new();
    loop {
        match read_record(&mut reader) {
            Ok(Some(sample)) => samples.push(sample),
            Ok(None) => break,
            Err(error) => return Err(error),
        }
    }
    Ok(samples)
}

pub fn write_header(writer: &mut impl Write) -> io::Result<()> {
    writer.write_all(MAGIC)
}

pub fn write_record(writer: &mut impl Write, sample: &BinarySample) -> io::Result<()> {
    writer.write_all(&sample.black_bits.to_le_bytes())?;
    writer.write_all(&sample.white_bits.to_le_bytes())?;
    writer.write_all(&[u8::from(sample.side_to_move_is_black)])?;
    writer.write_all(&sample.ply.to_le_bytes())?;
    writer.write_all(&sample.no_progress_plies.to_le_bytes())?;
    writer.write_all(&sample.score.to_le_bytes())?;
    writer.write_all(&sample.clipped_score.to_le_bytes())?;
    writer.write_all(&sample.result.to_le_bytes())?;
    writer.write_all(&sample.result_bucket.to_le_bytes())?;
    writer.write_all(&sample.completed_depth.to_le_bytes())?;
    writer.write_all(&sample.nodes.to_le_bytes())?;
    writer.write_all(&sample.elapsed_ms.to_le_bytes())?;
    writer.write_all(&[u8::from(sample.caused_ejection)])?;
    writer.write_all(&sample.occurrence_count.to_le_bytes())?;
    writer.write_all(&sample.sample_weight.to_le_bytes())?;
    Ok(())
}

fn read_record(reader: &mut impl Read) -> io::Result<Option<BinarySample>> {
    let Some(black_bits) = read_u64_or_eof(reader)? else {
        return Ok(None);
    };
    Ok(Some(BinarySample {
        black_bits,
        white_bits: read_u64(reader)?,
        side_to_move_is_black: read_u8(reader)? != 0,
        ply: read_f32(reader)?,
        no_progress_plies: read_f32(reader)?,
        score: read_f32(reader)?,
        clipped_score: read_f32(reader)?,
        result: read_f32(reader)?,
        result_bucket: read_i32(reader)?,
        completed_depth: read_f32(reader)?,
        nodes: read_u64(reader)?,
        elapsed_ms: read_u64(reader)?,
        caused_ejection: read_u8(reader)? != 0,
        occurrence_count: read_u32(reader)?,
        sample_weight: read_f32(reader)?,
    }))
}

fn read_u64_or_eof(reader: &mut impl Read) -> io::Result<Option<u64>> {
    let mut bytes = [0_u8; 8];
    match reader.read_exact(&mut bytes) {
        Ok(()) => Ok(Some(u64::from_le_bytes(bytes))),
        Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => Ok(None),
        Err(error) => Err(error),
    }
}

fn read_u64(reader: &mut impl Read) -> io::Result<u64> {
    let mut bytes = [0_u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_u32(reader: &mut impl Read) -> io::Result<u32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_i32(reader: &mut impl Read) -> io::Result<i32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(i32::from_le_bytes(bytes))
}

fn read_f32(reader: &mut impl Read) -> io::Result<f32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(f32::from_le_bytes(bytes))
}

fn read_u8(reader: &mut impl Read) -> io::Result<u8> {
    let mut bytes = [0_u8; 1];
    reader.read_exact(&mut bytes)?;
    Ok(bytes[0])
}
