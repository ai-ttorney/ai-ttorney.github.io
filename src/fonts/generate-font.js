import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';


const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);


const fontPath = path.join(__dirname, 'fonts', 'Roboto-Regular.ttf');
const fontData = fs.readFileSync(fontPath);


const base64Font = fontData.toString('base64');


const fontFileContent = `export default '${base64Font}';`;


const outputPath = path.join(__dirname, 'fonts', 'Roboto-Regular.js');
fs.writeFileSync(outputPath, fontFileContent);

console.log('Font file generated successfully at:', outputPath);
