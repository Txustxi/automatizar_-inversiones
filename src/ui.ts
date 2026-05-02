import { lessons, questions } from './data';
import { AppStore } from './store';

export function renderApp(root: HTMLElement, store: AppStore) {
  root.replaceChildren();

  const main = document.createElement('main');
  const title = document.createElement('h1');
  title.textContent = 'OpoRentable';

  const streak = document.createElement('p');
  streak.setAttribute('aria-live', 'polite');
  streak.textContent = `Racha: ${store.progress.streak} días`;

  const premium = document.createElement('button');
  premium.textContent = store.progress.premium ? 'Premium activo' : 'Activar premium (demo)';
  premium.disabled = store.progress.premium;
  premium.addEventListener('click', () => {
    store.unlockPremium();
    renderApp(root, store);
  });

  const lessonList = document.createElement('section');
  const lessonTitle = document.createElement('h2');
  lessonTitle.textContent = 'Lecciones';
  lessonList.appendChild(lessonTitle);

  lessons.forEach((lesson) => {
    const card = document.createElement('article');
    const h3 = document.createElement('h3');
    h3.textContent = lesson.title;
    const p = document.createElement('p');
    p.textContent = lesson.body;
    const done = document.createElement('button');
    const isDone = store.progress.lessonDone.includes(lesson.id);
    done.textContent = isDone ? 'Completada' : 'Marcar completada';
    done.disabled = isDone;
    done.addEventListener('click', () => {
      store.completeLesson(lesson.id);
      renderApp(root, store);
    });
    card.append(h3, p, done);
    lessonList.appendChild(card);
  });

  const exam = document.createElement('section');
  const examBtn = document.createElement('button');
  examBtn.textContent = 'Hacer mini test';
  examBtn.addEventListener('click', () => {
    let score = 0;
    questions.forEach((q) => {
      const answer = prompt(`${q.text}\n${q.options.map((o, i) => `${i + 1}. ${o}`).join('\n')}`);
      if (answer && Number(answer) - 1 === q.correct) score += 1;
    });
    store.addExamScore(score);
    renderApp(root, store);
  });
  exam.appendChild(examBtn);

  const history = document.createElement('ul');
  store.progress.examHistory.forEach((item) => {
    const li = document.createElement('li');
    li.textContent = `${new Date(item.at).toLocaleString('es-ES')}: ${item.score}/${questions.length}`;
    history.appendChild(li);
  });

  main.append(title, streak, premium, lessonList, exam, history);
  root.appendChild(main);
}
