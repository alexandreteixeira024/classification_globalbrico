import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const sourcePath = "data/emails_classificacao.xlsx";
const outputPath = "outputs/email-classification-polished/emails_classificacao.xlsx";

function decodeEntities(value) {
  return value
    .replaceAll("&nbsp;", " ")
    .replaceAll("&amp;", "&")
    .replaceAll("&lt;", "<")
    .replaceAll("&gt;", ">")
    .replaceAll("&quot;", '"')
    .replaceAll("&#39;", "'");
}

function readableText(data) {
  let value = data.text || data.body || data.html || "";
  if (typeof value !== "string") value = String(value);
  if (/<[^>]+>/.test(value)) {
    value = value
      .replace(/<br\s*\/?>|<\/p>|<\/div>|<\/li>|<\/tr>/gi, "\n")
      .replace(/<[^>]+>/g, " ");
    value = decodeEntities(value);
  }
  return value.replace(/\s+/g, " ").trim().slice(0, 32767);
}

async function loadEmails(directory) {
  const byUid = new Map();
  let names = [];
  try {
    names = await fs.readdir(directory);
  } catch {
    return byUid;
  }
  for (const name of names.filter((item) => item.endsWith(".json") && item !== "summary.json")) {
    try {
      const data = JSON.parse(await fs.readFile(path.join(directory, name), "utf8"));
      const uid = String(data.uid || data.id || "").replace(/\.0$/, "").trim();
      if (uid && !byUid.has(uid)) byUid.set(uid, data);
    } catch {
      // Ignore malformed source files; missing UIDs are checked below.
    }
  }
  return byUid;
}

const emails = await loadEmails("data/globalbrico_emails");
const fallback = await loadEmails("data/extracted_emails");
for (const [uid, data] of fallback) if (!emails.has(uid)) emails.set(uid, data);

const input = await FileBlob.load(sourcePath);
const workbook = await SpreadsheetFile.importXlsx(input);
const sheet = workbook.worksheets.getItem("Revisão");
const used = sheet.getUsedRange();
const rowCount = used.values.length;
const uidValues = sheet.getRange(`A2:A${rowCount}`).values;
const missing = [];
const textValues = uidValues.map(([rawUid]) => {
  const uid = String(rawUid ?? "").replace(/\.0$/, "").trim();
  const email = emails.get(uid);
  if (!email) {
    missing.push(uid);
    return [""];
  }
  return [readableText(email)];
});
if (missing.length) throw new Error(`Sem JSON para os UIDs: ${missing.join(", ")}`);

sheet.getRange("E1").values = [["Texto completo"]];
sheet.getRange(`E2:E${rowCount}`).values = textValues;
sheet.showGridLines = false;
sheet.freezePanes.freezeRows(1);
sheet.freezePanes.freezeColumns(4);

const allData = sheet.getRange(`A1:N${rowCount}`);
allData.format.font = { name: "Arial", size: 10, color: "#1F2937" };
allData.format.verticalAlignment = "top";

const header = sheet.getRange("A1:N1");
header.format = {
  fill: "#1F4E78",
  font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
  borders: { preset: "all", style: "thin", color: "#FFFFFF" },
};
header.format.rowHeight = 34;

sheet.getRange(`A2:N${rowCount}`).format.rowHeight = 88;
sheet.getRange(`A2:A${rowCount}`).format.horizontalAlignment = "center";
sheet.getRange(`A2:B${rowCount}`).format.verticalAlignment = "middle";
sheet.getRange(`C2:E${rowCount}`).format.wrapText = true;
sheet.getRange(`F2:F${rowCount}`).format = {
  fill: "#FFF2CC",
  font: { name: "Arial", size: 10, bold: true, color: "#7F6000" },
  horizontalAlignment: "center",
  verticalAlignment: "middle",
  wrapText: true,
  borders: { preset: "all", style: "thin", color: "#D6B656" },
};
sheet.getRange(`G2:N${rowCount}`).format.wrapText = true;

const widths = {
  A: 11, B: 18, C: 28, D: 38, E: 58, F: 24, G: 22,
  H: 22, I: 15, J: 22, K: 24, L: 13, M: 24, N: 34,
};
for (const [column, width] of Object.entries(widths)) {
  sheet.getRange(`${column}1:${column}${rowCount}`).format.columnWidth = width;
}

const tables = sheet.tables.items;
if (tables.length) {
  tables[0].style = "TableStyleMedium2";
  tables[0].showBandedColumns = false;
  tables[0].showFilterButton = true;
}

workbook.recalculate();
const check = await workbook.inspect({
  kind: "table",
  range: `Revisão!A1:F${rowCount}`,
  include: "values,formulas",
  tableMaxRows: 8,
  tableMaxCols: 6,
  tableMaxCellChars: 180,
  maxChars: 10000,
});
console.log(check.ndjson);
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

const preview = await workbook.render({
  sheetName: "Revisão",
  range: "A1:F10",
  scale: 1,
  format: "png",
});
await fs.writeFile(".codex-artifacts/email-labels/full_text.png", new Uint8Array(await preview.arrayBuffer()));

await fs.mkdir(path.dirname(outputPath), { recursive: true });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(JSON.stringify({ outputPath, rowsUpdated: textValues.length }));
