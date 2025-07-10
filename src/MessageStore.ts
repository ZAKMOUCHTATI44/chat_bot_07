class ExpiringMessageHistory {
  private store: Map<string, { messages: any[]; timestamp: number }> =
    new Map();
  private ttl: number;

  constructor(ttlMs: number) {
    this.ttl = ttlMs;
  }

  getMessages(sessionId: string) {
    const data = this.store.get(sessionId);
    if (!data || Date.now() - data.timestamp > this.ttl) {
      this.store.delete(sessionId); // Expired
      return [];
    }
    return data.messages;
  }

  addMessage(sessionId: string, message: any) {
    const existing = this.getMessages(sessionId);
    const updatedMessages = [...existing, message];
    this.store.set(sessionId, {
      messages: updatedMessages,
      timestamp: Date.now(),
    });
  }

  clearExpired() {
    const now = Date.now();
    for (const [sessionId, data] of this.store.entries()) {
      if (now - data.timestamp > this.ttl) {
        this.store.delete(sessionId);
      }
    }
  }
}
