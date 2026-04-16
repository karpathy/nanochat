export interface SlashResult {
  handled: boolean;
  consoleMessage?: string;
  clear?: boolean;
  setTemperature?: number;
  setTopK?: number;
}

export function parseSlashCommand(
  raw: string,
  state: { temperature: number; topK: number },
): SlashResult {
  const line = raw.trim();
  if (!line.startsWith('/')) return { handled: false };

  const [cmd, arg] = line.split(/\s+/);
  switch (cmd.toLowerCase()) {
    case '/temperature': {
      if (arg === undefined) {
        return { handled: true, consoleMessage: `Current temperature: ${state.temperature}` };
      }
      const t = parseFloat(arg);
      if (isNaN(t) || t < 0 || t > 2) {
        return { handled: true, consoleMessage: 'Invalid temperature. Must be between 0.0 and 2.0' };
      }
      return { handled: true, setTemperature: t, consoleMessage: `Temperature set to ${t}` };
    }
    case '/topk': {
      if (arg === undefined) {
        return { handled: true, consoleMessage: `Current top-k: ${state.topK}` };
      }
      const k = parseInt(arg, 10);
      if (isNaN(k) || k < 1 || k > 200) {
        return { handled: true, consoleMessage: 'Invalid top-k. Must be between 1 and 200' };
      }
      return { handled: true, setTopK: k, consoleMessage: `Top-k set to ${k}` };
    }
    case '/clear':
      return { handled: true, clear: true };
    case '/help':
      return {
        handled: true,
        consoleMessage:
          'Commands:\n/temperature [0-2] — sampling temperature\n/topk [1-200] — top-k sampling\n/clear — clear conversation\n/help — show this help',
      };
    default:
      return { handled: true, consoleMessage: `Unknown command: ${cmd}` };
  }
}
