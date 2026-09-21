import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const input = await FileBlob.load("data/emails_classificacao.xlsx");
const workbook = await SpreadsheetFile.importXlsx(input);
const overview = await workbook.inspect({
  kind: "workbook,sheet,table",
  maxChars: 8000,
  tableMaxRows: 8,
  tableMaxCols: 16,
  tableMaxCellChars: 100,
});
console.log(overview.ndjson);
const preview = await workbook.render({
  sheetName: "Revisão",
  range: "A1:N20",
  scale: 1,
  format: "png",
});
await fs.writeFile(".codex-artifacts/email-labels/before.png", new Uint8Array(await preview.arrayBuffer()));
