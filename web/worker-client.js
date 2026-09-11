// Owns one worker generation. Disposing or failing it settles every readiness wait.
export class SearchWorkerClient {
  constructor(url, onMessage, onFailure, WorkerType = Worker) {
    this.status = 'initializing';
    this.ready = new Promise((resolve, reject) => {
      this.resolveReady = resolve;
      this.rejectReady = reject;
    });
    // A worker can fail before a caller awaits ready. Keep the original rejected
    // promise for callers, but avoid an unhandled-rejection notification.
    this.ready.catch(() => {});
    try {
      this.worker = new WorkerType(url, { type: 'module' });
      this.worker.addEventListener('message', (event) => {
        if (this.status === 'disposed' || this.status === 'failed') return;
        if (event.data.type === 'ready') {
          this.status = 'ready';
          this.resolveReady();
        } else if (event.data.type === 'init-error') {
          this.fail(new Error(event.data.error || 'Worker initialization failed'), onFailure);
        } else {
          onMessage(event);
        }
      });
      this.worker.addEventListener('error', (event) => {
        this.fail(new Error(event.message || 'Worker error'), onFailure);
      });
      this.worker.addEventListener('messageerror', () => {
        this.fail(new Error('Could not decode worker response'), onFailure);
      });
      this.worker.postMessage({ type: 'init' });
    } catch (error) {
      this.fail(error, onFailure);
    }
  }

  fail(error, onFailure) {
    if (this.status === 'disposed' || this.status === 'failed') return;
    this.status = 'failed';
    this.rejectReady(error);
    this.worker?.terminate();
    onFailure(error);
  }

  postMessage(message) {
    if (this.status !== 'ready') throw new Error(`Worker is ${this.status}`);
    this.worker.postMessage(message);
  }

  terminate() {
    if (this.status === 'disposed') return;
    this.status = 'disposed';
    this.rejectReady(new Error('Worker replaced or disposed'));
    this.worker?.terminate();
  }
}
