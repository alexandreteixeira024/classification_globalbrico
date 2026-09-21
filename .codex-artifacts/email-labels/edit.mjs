import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const sourcePath = "data/emails_classificacao.xlsx";
const outputPath = "outputs/email-labels/emails_classificacao.xlsx";
const input = await FileBlob.load(sourcePath);
const workbook = await SpreadsheetFile.importXlsx(input);
const sheet = workbook.worksheets.getItem("Revisão");

sheet.dataValidations.add({
  range: "F2:F1000",
  rule: {
    type: "list",
    values: ["Pedido de Informação", "Pedido de Encomenda", "SPAM"],
  },
});

workbook.recalculate();

const keyRange = await workbook.inspect({
  kind: "table",
  range: "Revisão!A1:N29",
  include: "values,formulas",
  tableMaxRows: 29,
  tableMaxCols: 14,
  maxChars: 12000,
});
console.log(keyRange.ndjson);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

const preview = await workbook.render({
  sheetName: "Revisão",
  range: "A1:N20",
  scale: 1,
  format: "png",
});
await fs.writeFile(".codex-artifacts/email-labels/after.png", new Uint8Array(await preview.arrayBuffer()));

await fs.mkdir("outputs/email-labels", { recursive: true });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);

console.log(JSON.stringify({ validationRange: "Revisão!F2:F1000", outputPath }));
