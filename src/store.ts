import type { Progress } from './types';

const KEY = 'oporentable.progress.v1';

const today = () => new Date().toISOString().slice(0, 10);

export class AppStore {
  progress: Progress;

  constructor() {
    const saved = localStorage.getItem(KEY);
    this.progress = saved
      ? JSON.parse(saved)
      : { streak: 0, lastSessionDate: null, lessonDone: [], examHistory: [], premium: false };
    this.touchStreak();
  }

  private persist() {
    localStorage.setItem(KEY, JSON.stringify(this.progress));
  }

  touchStreak() {
    const now = today();
    const last = this.progress.lastSessionDate;
    if (last === now) return;
    if (!last) this.progress.streak = 1;
    else {
      const diff = Math.round((Date.parse(now) - Date.parse(last)) / 86400000);
      this.progress.streak = diff === 1 ? this.progress.streak + 1 : 1;
    }
    this.progress.lastSessionDate = now;
    this.persist();
  }

  completeLesson(id: string) {
    if (!this.progress.lessonDone.includes(id)) this.progress.lessonDone.push(id);
    this.persist();
  }

  addExamScore(score: number) {
    this.progress.examHistory.unshift({ score, at: new Date().toISOString() });
    this.progress.examHistory = this.progress.examHistory.slice(0, 20);
    this.persist();
  }

  unlockPremium() {
    this.progress.premium = true;
    this.persist();
  }
}
