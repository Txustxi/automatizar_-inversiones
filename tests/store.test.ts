import { describe, expect, it, vi, beforeEach } from 'vitest';
import { AppStore } from '../src/store';

describe('AppStore', () => {
  beforeEach(() => {
    localStorage.clear();
    vi.useRealTimers();
  });

  it('inicializa racha en 1', () => {
    const s = new AppStore();
    expect(s.progress.streak).toBe(1);
  });
});
