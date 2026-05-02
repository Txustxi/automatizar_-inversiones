import { registerSW } from './sw';
import { AppStore } from './store';
import { renderApp } from './ui';

const store = new AppStore();
const mount = document.getElementById('app');

if (!mount) throw new Error('No se encontró #app');

renderApp(mount, store);
registerSW();
