// MongoDB Schema for AMATO Production
// Clickstream Data

// Create database
use amato_production;

// Create collections with validation schemas

// Sessions collection
db.createCollection("sessions", {
   validator: {
      $jsonSchema: {
         bsonType: "object",
         required: ["session_id", "customer_id", "session_start"],
         properties: {
            session_id: {
               bsonType: "string",
               description: "must be a string and is required"
            },
            customer_id: {
               bsonType: "string",
               description: "must be a string and is required"
            },
            session_start: {
               bsonType: "date",
               description: "must be a date and is required"
            },
            session_end: {
               bsonType: "date"
            },
            duration_seconds: {
               bsonType: "int"
            },
            device_type: {
               bsonType: "string"
            },
            browser: {
               bsonType: "string"
            },
            os: {
               bsonType: "string"
            },
            ip_address: {
               bsonType: "string"
            },
            user_agent: {
               bsonType: "string"
            },
            referrer_url: {
               bsonType: "string"
            },
            landing_page: {
               bsonType: "string"
            },
            exit_page: {
               bsonType: "string"
            },
            pages_viewed: {
               bsonType: "int"
            },
            created_at: {
               bsonType: "date"
            }
         }
      }
   }
});

// Page views collection
db.createCollection("page_views", {
   validator: {
      $jsonSchema: {
         bsonType: "object",
         required: ["view_id", "session_id", "page_url", "timestamp"],
         properties: {
            view_id: {
               bsonType: "string",
               description: "must be a string and is required"
            },
            session_id: {
               bsonType: "string",
               description: "must be a string and is required"
            },
            customer_id: {
               bsonType: "string"
            },
            page_url: {
               bsonType: "string",
               description: "must be a string and is required"
            },
            page_title: {
               bsonType: "string"
            },
            page_category: {
               bsonType: "string"
            },
            page_type: {
               bsonType: "string"
            },
            time_on_page: {
               bsonType: "int"
            },
            scroll_depth: {
               bsonType: "int"
            },
            timestamp: {
               bsonType: "date",
               description: "must be a date and is required"
            },
            created_at: {
               bsonType: "date"
            }
         }
      }
   }
});

// Events collection
db.createCollection("events", {
   validator: {
      $jsonSchema: {
         bsonType: "object",
         required: ["event_id", "session_id", "event_type", "timestamp"],
         properties: {
            event_id: {
               bsonType: "string",
               description: "must be a string and is required"
            },
            session_id: {
               bsonType: "string",
               description: "must be a string and is required"
            },
            customer_id: {
               bsonType: "string"
            },
            event_type: {
               bsonType: "string",
               description: "must be a string and is required"
            },
            event_name: {
               bsonType: "string"
            },
            event_data: {
               bsonType: "object"
            },
            timestamp: {
               bsonType: "date",
               description: "must be a date and is required"
            },
            created_at: {
               bsonType: "date"
            }
         }
      }
   }
});

// Product interactions collection
db.createCollection("product_interactions", {
   validator: {
      $jsonSchema: {
         bsonType: "object",
         required: ["interaction_id", "session_id", "product_id", "interaction_type", "timestamp"],
         properties: {
            interaction_id: {
               bsonType: "string",
               description: "must be a string and is required"
            },
            session_id: {
               bsonType: "string",
               description: "must be a string and is required"
            },
            customer_id: {
               bsonType: "string"
            },
            product_id: {
               bsonType: "string",
               description: "must be a string and is required"
            },
            interaction_type: {
               bsonType: "string",
               description: "must be a string and is required"
            },
            interaction_data: {
               bsonType: "object"
            },
            timestamp: {
               bsonType: "date",
               description: "must be a date and is required"
            },
            created_at: {
               bsonType: "date"
            }
         }
      }
   }
});

// Search queries collection
db.createCollection("search_queries", {
   validator: {
      $jsonSchema: {
         bsonType: "object",
         required: ["query_id", "session_id", "search_term", "timestamp"],
         properties: {
            query_id: {
               bsonType: "string",
               description: "must be a string and is required"
            },
            session_id: {
               bsonType: "string",
               description: "must be a string and is required"
            },
            customer_id: {
               bsonType: "string"
            },
            search_term: {
               bsonType: "string",
               description: "must be a string and is required"
            },
            search_results_count: {
               bsonType: "int"
            },
            clicked_result_position: {
               bsonType: "int"
            },
            timestamp: {
               bsonType: "date",
               description: "must be a date and is required"
            },
            created_at: {
               bsonType: "date"
            }
         }
      }
   }
});

// Create indexes for better performance
db.sessions.createIndex({ "session_id": 1 }, { unique: true });
db.sessions.createIndex({ "customer_id": 1 });
db.sessions.createIndex({ "session_start": 1 });
db.sessions.createIndex({ "session_id": 1, "customer_id": 1 });

db.page_views.createIndex({ "view_id": 1 }, { unique: true });
db.page_views.createIndex({ "session_id": 1 });
db.page_views.createIndex({ "customer_id": 1 });
db.page_views.createIndex({ "timestamp": 1 });
db.page_views.createIndex({ "session_id": 1, "timestamp": 1 });

db.events.createIndex({ "event_id": 1 }, { unique: true });
db.events.createIndex({ "session_id": 1 });
db.events.createIndex({ "customer_id": 1 });
db.events.createIndex({ "event_type": 1 });
db.events.createIndex({ "timestamp": 1 });
db.events.createIndex({ "session_id": 1, "event_type": 1 });

db.product_interactions.createIndex({ "interaction_id": 1 }, { unique: true });
db.product_interactions.createIndex({ "session_id": 1 });
db.product_interactions.createIndex({ "customer_id": 1 });
db.product_interactions.createIndex({ "product_id": 1 });
db.product_interactions.createIndex({ "interaction_type": 1 });
db.product_interactions.createIndex({ "timestamp": 1 });

db.search_queries.createIndex({ "query_id": 1 }, { unique: true });
db.search_queries.createIndex({ "session_id": 1 });
db.search_queries.createIndex({ "customer_id": 1 });
db.search_queries.createIndex({ "search_term": 1 });
db.search_queries.createIndex({ "timestamp": 1 });

print("MongoDB schema created successfully!");
