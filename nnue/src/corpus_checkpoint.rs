//! A committed append-only prefix for records, deduplication keys and shard state.
//! Writers must flush and sync both data files before publishing this checkpoint.
use anyhow::{Context, Result, bail};
use serde::{Serialize, de::DeserializeOwned};
use std::{
    fs,
    io::{Seek, SeekFrom, Write},
    path::Path,
};

#[derive(serde::Serialize, serde::Deserialize)]
struct Checkpoint<T> {
    version: u32,
    accepted_bytes: u64,
    seen_bytes: u64,
    signatures: T,
}

pub(super) fn recover<T: DeserializeOwned>(
    path: &Path,
    accepted: &Path,
    seen: &Path,
) -> Result<Option<T>> {
    if !path.exists() {
        return Ok(None);
    }
    let checkpoint: Checkpoint<T> = serde_json::from_slice(&fs::read(path)?)?;
    if checkpoint.version != 1 {
        bail!("unsupported corpus checkpoint version");
    }
    // Verify both before touching either. A missing/short committed file is a
    // hard error; silently rebuilding it could turn lost records into exclusions.
    for (file, size) in [
        (accepted, checkpoint.accepted_bytes),
        (seen, checkpoint.seen_bytes),
    ] {
        if fs::metadata(file)
            .with_context(|| format!("missing committed corpus file {}", file.display()))?
            .len()
            < size
        {
            bail!("{} is shorter than its committed prefix", file.display());
        }
    }
    for (file, size) in [
        (accepted, checkpoint.accepted_bytes),
        (seen, checkpoint.seen_bytes),
    ] {
        let mut handle = fs::OpenOptions::new().read(true).write(true).open(file)?;
        if handle.metadata()?.len() > size {
            // Preserve uncommitted bytes before truncation for diagnosis/recovery.
            let stamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos();
            let backup = file.with_extension(format!("uncommitted-{stamp}"));
            handle.seek(SeekFrom::Start(size))?;
            let mut tail = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(backup)?;
            std::io::copy(&mut handle, &mut tail)?;
            tail.sync_all()?;
            #[cfg(unix)]
            fs::File::open(
                file.parent()
                    .context("corpus file needs parent directory")?,
            )?
            .sync_all()?;
            handle.set_len(size)?;
            handle.sync_all()?;
        }
    }
    Ok(Some(checkpoint.signatures))
}

pub(super) fn commit<T: Serialize>(
    path: &Path,
    accepted: &Path,
    seen: &Path,
    signatures: &T,
) -> Result<()> {
    let data = Checkpoint {
        version: 1,
        accepted_bytes: fs::metadata(accepted)?.len(),
        seen_bytes: fs::metadata(seen)?.len(),
        signatures,
    };
    let temporary = path.with_extension("pending");
    let mut file = fs::File::create(&temporary)?;
    serde_json::to_writer(&mut file, &data)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    fs::rename(temporary, path)?;
    #[cfg(unix)]
    fs::File::open(path.parent().context("checkpoint needs parent directory")?)?.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn interrupted_append_restores_both_committed_prefixes() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "steinbeisser-checkpoint-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        fs::create_dir(&root)?;
        let a = root.join("accepted");
        let s = root.join("seen");
        let c = root.join("checkpoint");
        fs::write(&a, b"records")?;
        fs::write(&s, b"keys")?;
        commit(&c, &a, &s, &vec![1u32])?;
        fs::OpenOptions::new()
            .append(true)
            .open(&a)?
            .write_all(b"partial")?;
        fs::OpenOptions::new()
            .append(true)
            .open(&s)?
            .write_all(b"partial-key")?;
        assert_eq!(recover::<Vec<u32>>(&c, &a, &s)?, Some(vec![1]));
        assert_eq!(fs::read(&a)?, b"records");
        assert_eq!(fs::read(&s)?, b"keys");
        assert_eq!(recover::<Vec<u32>>(&c, &a, &s)?, Some(vec![1]));
        fs::write(&a, b"short")?;
        fs::write(&s, b"keys-and-an-uncommitted-tail")?;
        assert!(recover::<Vec<u32>>(&c, &a, &s).is_err());
        assert_eq!(fs::read(&s)?, b"keys-and-an-uncommitted-tail");
        fs::write(&a, b"records-and-an-uncommitted-tail")?;
        fs::write(&s, b"key")?;
        assert!(recover::<Vec<u32>>(&c, &a, &s).is_err());
        assert_eq!(fs::read(&a)?, b"records-and-an-uncommitted-tail");
        fs::write(&c, b"{\"version\":1,")?;
        assert!(recover::<Vec<u32>>(&c, &a, &s).is_err());
        assert_eq!(fs::read(&a)?, b"records-and-an-uncommitted-tail");
        fs::remove_dir_all(root)?;
        Ok(())
    }
}
