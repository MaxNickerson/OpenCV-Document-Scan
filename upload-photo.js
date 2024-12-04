import fs from 'fs';
import path from 'path';

export const config = {
    api: {
        bodyParser: false, // Disable default body parser
    },
};

const handler = async (req, res) => {
    if (req.method === 'POST') {
        const chunks = [];
        req.on('data', (chunk) => chunks.push(chunk));
        req.on('end', () => {
            const buffer = Buffer.concat(chunks);
            const filePath = path.join(process.cwd(), 'uploads', `photo-${Date.now()}.png`);
            fs.writeFileSync(filePath, buffer);
            res.status(200).json({ message: 'Photo uploaded successfully', filePath });
        });
    } else {
        res.status(405).json({ message: 'Method Not Allowed' });
    }
};

export default handler;
