import * as admin from "firebase-admin";
import * as fs from "fs";
import * as path from "path";

export class FriendsDataMigrator {
  private db: admin.firestore.Firestore;

  constructor() {
    // Initialize Firebase Admin only if not already initialized
    if (admin.apps.length === 0) {
      // Try to load service account key first
      try {
        const serviceAccount = require("../../../conversation-practice-f2199-firebase-adminsdk-fbsvc-1e1af80c9c.json");
        admin.initializeApp({
          credential: admin.credential.cert(serviceAccount),
          projectId: "conversation-practice-f2199",
        });
      } catch (error) {
        // Fallback to default initialization
        admin.initializeApp({
          projectId: "conversation-practice-f2199",
        });
      }
    }
    this.db = admin.firestore();
    
    // Use Firestore emulator if available
    if (process.env.FIRESTORE_EMULATOR_HOST) {
      this.db.settings({
        host: process.env.FIRESTORE_EMULATOR_HOST,
        ssl: false,
      });
    }
  }

  /**
   * Main migration method for all data_ready files
   */
  async migrateAllDataReadyFiles(): Promise<void> {
    console.log("Starting data_ready files migration to Firestore...");
    
    try {
      // Migrate all JSONL files from data_ready folder
      await this.migrateAllDataReadyJSONL();
      
      console.log("✅ All data_ready files migrated successfully!");
    } catch (error) {
      console.error("❌ Migration failed:", error);
      throw error;
    }
  }

  /**
   * Migrate all JSONL files from data_ready folder
   */
  async migrateAllDataReadyJSONL(): Promise<void> {
    console.log("📁 Processing all data_ready JSONL files...");
    
    try {
      const dataReadyDir = path.join(__dirname, "../../../data_ready");
      
      // Check if directory exists
      if (!fs.existsSync(dataReadyDir)) {
        throw new Error("data_ready directory not found");
      }

      const jsonlFiles = fs
        .readdirSync(dataReadyDir)
        .filter((file) => file.endsWith(".jsonl"));

      console.log(`Found ${jsonlFiles.length} JSONL files to process`);

      let totalRecords = 0;
      const timestamp = admin.firestore.Timestamp.now();

      for (const file of jsonlFiles) {
        console.log(`📄 Processing ${file}...`);
        
        const filePath = path.join(dataReadyDir, file);
        const content = fs.readFileSync(filePath, "utf8");
        const lines = content.trim().split("\n");

        // Process file in batches
        const batchSize = 400; // Firestore batch limit is 500, using 400 for safety
        let batch = this.db.batch();
        let batchCount = 0;

        for (const line of lines) {
          try {
            const data = JSON.parse(line);
            
            // Determine collection and document ID
            const { collection, docId } = this.getCollectionAndDocId(data, file);
            
            // Add metadata
            const documentData = {
              ...data,
              created_at: timestamp,
              updated_at: timestamp,
              source_file: file,
            };

            const docRef = this.db.collection(collection).doc(docId);
            batch.set(docRef, documentData);
            
            batchCount++;
            totalRecords++;

            // Commit batch if limit reached
            if (batchCount >= batchSize) {
              await batch.commit();
              console.log(`  Committed batch of ${batchCount} records`);
              batch = this.db.batch();
              batchCount = 0;
            }
          } catch (parseError) {
            console.warn(`⚠️  Skipping invalid JSON line in ${file}:`, parseError);
          }
        }

        // Commit remaining records in batch
        if (batchCount > 0) {
          await batch.commit();
          console.log(`  Committed final batch of ${batchCount} records for ${file}`);
        }

        console.log(`✅ Processed ${file} - ${lines.length} records`);
      }

      console.log(`✅ Successfully migrated ${totalRecords} total records from data_ready`);
    } catch (error) {
      console.error("❌ Error migrating data_ready files:", error);
      throw error;
    }
  }

  /**
   * Determine Firestore collection and document ID based on data structure and filename
   */
  private getCollectionAndDocId(data: any, filename: string): { collection: string; docId: string } {
    // If it's plots data
    if (filename.includes("plots_upsert.jsonl")) {
      return {
        collection: "friends_plots",
        docId: data.id || data.episode_id || `plot_${Date.now()}_${Math.random()}`
      };
    }

    // For episode scene data (most files)
    if (data.metadata && data.metadata.chunk_type === "scene") {
      return {
        collection: "friends_scenes",
        docId: data.id || data.metadata.scene_id || `scene_${Date.now()}_${Math.random()}`
      };
    }

    // For any other structured data with chunk_type
    if (data.metadata && data.metadata.chunk_type) {
      const collectionMap: { [key: string]: string } = {
        'plot': 'friends_plots',
        'scene': 'friends_scenes',
        'episode': 'friends_episodes',
        'character': 'friends_characters'
      };
      
      const collection = collectionMap[data.metadata.chunk_type] || 'friends_misc';
      return {
        collection,
        docId: data.id || `${data.metadata.chunk_type}_${Date.now()}_${Math.random()}`
      };
    }

    // For episode data (general)
    if (data.episode_id || data.id?.match(/S\d+E\d+/)) {
      return {
        collection: "friends_episodes_data",
        docId: data.id || data.episode_id || `episode_${Date.now()}_${Math.random()}`
      };
    }

    // Default fallback
    return {
      collection: "friends_data",
      docId: data.id || `data_${Date.now()}_${Math.random()}`
    };
  }

  /**
   * Clear all Friends collections (use with caution)
   */
  async clearAllCollections(): Promise<void> {
    console.log("🗑️  Clearing all Friends collections...");
    
    const collections = [
      "friends_scenes",
      "friends_plots", 
      "friends_episodes_data",
      "friends_data",
      "friends_misc"
    ];

    for (const collectionName of collections) {
      console.log(`Clearing ${collectionName}...`);
      await this.clearCollection(collectionName);
    }

    console.log("✅ All collections cleared");
  }

  /**
   * Clear a specific collection
   */
  private async clearCollection(collectionName: string): Promise<void> {
    const batchSize = 100;
    let deleted = 0;

    while (true) {
      const snapshot = await this.db
        .collection(collectionName)
        .limit(batchSize)
        .get();

      if (snapshot.empty) {
        break;
      }

      const batch = this.db.batch();
      snapshot.docs.forEach((doc) => {
        batch.delete(doc.ref);
      });

      await batch.commit();
      deleted += snapshot.size;
      console.log(`  Deleted ${deleted} documents from ${collectionName}`);
    }
  }
}

// Run migration if this script is executed directly
if (require.main === module) {
  const migrator = new FriendsDataMigrator();
  
  // Add command line argument support
  const args = process.argv.slice(2);
  const shouldClear = args.includes('--clear');
  
  (async () => {
    try {
      if (shouldClear) {
        await migrator.clearAllCollections();
      }
      
      await migrator.migrateAllDataReadyFiles();
      console.log("Migration completed successfully!");
      process.exit(0);
    } catch (error) {
      console.error("Migration failed:", error);
      process.exit(1);
    }
  })();
}