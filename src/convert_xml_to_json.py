import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

RAW_DIR = Path("data/raw/xml")
OUT_DIR = Path("data/processed/json")
MAX_FILES = None

def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def safe_int(value, default=None):
    try:
        return int(value)
    except Exception:
        return default

def convert_file(xml_path: Path) -> dict:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    period = root.attrib.get("period")
    legislature = root.attrib.get("legislature")
    legislative_session = root.attrib.get("legislative_session")
    number = root.attrib.get("number")
    date = root.attrib.get("date")

    debate_id = f"{period}_{legislature}_{legislative_session}_{number}_{date}"

    utterances_out = []
    for idx, u in enumerate(root.findall("utterance"), start=1):
        speaker_id = safe_int(u.attrib.get("speaker-id"), default=None)
        speaker_name = u.attrib.get("speaker-name")
        speaker_party = u.attrib.get("speaker-party")
        speaker_string = u.attrib.get("speaker-string")
        speaker_role = u.attrib.get("speaker-role")
        page_start = safe_int(u.attrib.get("page-start"), default=None)

        text = normalize_text("".join(u.itertext()))

        speaker_is_generic = (
            speaker_id in (-1, None) or (speaker_name == "N/A" and speaker_party == "N/A")
        )

        utterances_out.append({
            "utterance_id": idx,
            "order": idx,
            "page_start": page_start,
            "speaker": {
                "id": speaker_id,
                "name": speaker_name,
                "party": speaker_party,
                "string": speaker_string,
                "role": speaker_role if speaker_role else None
            },
            "speaker_is_generic": speaker_is_generic,
            "text": text
        })

    return {
        "debate_meta": {
            "period": period,
            "legislature": legislature,
            "legislative_session": legislative_session,
            "number": number,
            "date": date
        },
        "debate_id": debate_id,
        "utterances": utterances_out
    }

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(RAW_DIR.glob("*.xml"))[:MAX_FILES]
    if not xml_files:
        print(f"Não encontrei XML em: {RAW_DIR.resolve()}")
        return

    ok = 0
    failed = 0

    for xml_path in xml_files:
        try:
            data = convert_file(xml_path)
            out_path = OUT_DIR / (xml_path.stem + ".json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            ok += 1
            print(f"[OK] {xml_path.name} -> {out_path.name} (utterances={len(data['utterances'])})")
        except Exception as e:
            failed += 1
            print(f"[FAIL] {xml_path.name}: {e}")

    print(f"\nConcluído (limite={MAX_FILES}). Sucesso: {ok} | Falhas: {failed}")

if __name__ == "__main__":
    main()