export type Lesson = { id: string; title: string; body: string };
export type Question = {
  id: string;
  lessonId: string;
  text: string;
  options: string[];
  correct: number;
};
export type Progress = {
  streak: number;
  lastSessionDate: string | null;
  lessonDone: string[];
  examHistory: { score: number; at: string }[];
  premium: boolean;
};
