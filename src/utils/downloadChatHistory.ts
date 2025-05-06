import { jsPDF } from 'jspdf';
import RobotoFont from '../fonts/Roboto-Regular.js';
import { Message} from '../types/chatTypes';

export const generateChatHistoryPDF = (messages: Message[], threadTitle: string, userName: string) => {
  const doc = new jsPDF({
    orientation: 'portrait',
    unit: 'mm',
    format: 'a4',
    putOnlyUsedFonts: true,
    floatPrecision: 16,
  });

  doc.addFileToVFS('Roboto-Regular.ttf', RobotoFont);
  doc.addFont('Roboto-Regular.ttf', 'Roboto', 'normal');
  doc.addFont('Roboto-Regular.ttf', 'Roboto', 'bold');
  doc.setLanguage('tr');

  doc.setFontSize(16);
  doc.setFont('Roboto', 'bold');
  doc.text(threadTitle, 20, 20);

  doc.setFontSize(10);
  doc.setFont('Roboto', 'normal');
  doc.text(`Downloaded on: ${new Date().toLocaleString('tr-TR')}`, 20, 30);

  let y = 40;
  const lineHeight = 7;
  const pageHeight = 280;
  const margin = 20;

  messages.forEach((msg: Message) => {
    if (y > pageHeight) {
      doc.addPage();
      y = margin;
    }

    doc.setFont('Roboto', 'bold');
    const role = msg.role === 'user' ? userName : 'AI-ttorney';
    doc.text(`${role} (${new Date(msg.timestamp).toLocaleString('tr-TR')}):`, margin, y);
    y += lineHeight;

    doc.setFont('Roboto', 'normal');
    const contentLines = doc.splitTextToSize(msg.content, 170);
    contentLines.forEach((line: string) => {
      if (y > pageHeight) {
        doc.addPage();
        y = margin;
      }
      doc.text(line, margin, y);
      y += lineHeight;
    });

    y += lineHeight;
  });

  doc.save(`${threadTitle}.pdf`);
};
