import type { Lesson, Question } from './types';

export const lessons: Lesson[] = [
  { id: 'l1', title: 'Constitución: derechos', body: 'Resumen corto de derechos fundamentales y garantías.' },
  { id: 'l2', title: 'Procedimiento administrativo', body: 'Fases, plazos y silencio administrativo.' }
];

export const questions: Question[] = [
  { id: 'q1', lessonId: 'l1', text: '¿Qué artículo reconoce la igualdad?', options: ['Art. 14', 'Art. 1', 'Art. 24'], correct: 0 },
  { id: 'q2', lessonId: 'l2', text: 'El silencio administrativo puede ser:', options: ['Sólo negativo', 'Positivo o negativo', 'Nunca aplica'], correct: 1 }
];
