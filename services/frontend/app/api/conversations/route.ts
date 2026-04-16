import { NextResponse } from 'next/server';

export const runtime = 'nodejs';

const day = 1000 * 60 * 60 * 24;
const now = () => Date.now();

export async function GET() {
  return NextResponse.json({
    conversations: [
      {
        id: 'mock-1',
        title: 'Why is samosa triangular?',
        updatedAt: now() - 1000 * 60 * 30,
      },
      {
        id: 'mock-2',
        title: 'Chai masala recipe',
        updatedAt: now() - day,
      },
      {
        id: 'mock-3',
        title: 'Explain transformers simply',
        updatedAt: now() - day * 4,
      },
      {
        id: 'mock-4',
        title: 'Monsoon pakora tips',
        updatedAt: now() - day * 15,
      },
    ],
  });
}
