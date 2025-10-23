// Test setup file
// Mock Firebase functions config
jest.mock('firebase-functions', () => ({
  config: jest.fn(() => ({
    openai: { api_key: 'test-openai-key' },
    pinecone: { api_key: 'test-pinecone-key' }
  })),
  https: {
    onRequest: jest.fn()
  }
}));

// Mock Firebase admin
jest.mock('firebase-admin', () => ({
  initializeApp: jest.fn(),
  firestore: jest.fn(() => ({
    collection: jest.fn(() => ({
      doc: jest.fn(() => ({
        get: jest.fn(),
        set: jest.fn(),
        update: jest.fn(),
        delete: jest.fn()
      })),
      where: jest.fn(() => ({
        orderBy: jest.fn(() => ({
          limit: jest.fn(() => ({
            get: jest.fn()
          }))
        }))
      }))
    }))
  })),
  Timestamp: {
    now: jest.fn(() => ({ toDate: () => new Date() }))
  }
}));