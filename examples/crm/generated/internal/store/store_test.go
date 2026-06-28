package store_test

import (
	"testing"

	lbug "github.com/LadybugDB/go-ladybug"
	"ladycrm/internal/store"
)

func TestInitAddContactList(t *testing.T) {
	db, err := lbug.OpenInMemoryDatabase(lbug.DefaultSystemConfig())
	if err != nil {
		t.Fatalf("open in-memory db: %v", err)
	}
	defer db.Close()
	conn, err := lbug.OpenConnection(db)
	if err != nil {
		t.Fatalf("open connection: %v", err)
	}
	defer conn.Close()

	// Run schema DDL.
	for _, ddl := range []string{
		`CREATE NODE TABLE IF NOT EXISTS Contact(id STRING, name STRING, email STRING, phone STRING, title STRING, created_at TIMESTAMP, PRIMARY KEY(id))`,
	} {
		if _, err := conn.Query(ddl); err != nil {
			t.Fatalf("ddl: %v", err)
		}
	}

	// Add a contact.
	cid, err := store.AddContact(conn, "Ada", "ada@x.io")
	if err != nil {
		t.Fatalf("add contact: %v", err)
	}
	if cid == "" {
		t.Fatal("expected non-empty contact id")
	}

	// List and assert.
	contacts, err := store.ListContacts(conn)
	if err != nil {
		t.Fatalf("list contacts: %v", err)
	}
	if len(contacts) != 1 {
		t.Fatalf("expected 1 contact, got %d", len(contacts))
	}
	c := contacts[0]
	if c.ID != cid {
		t.Errorf("id = %q, want %q", c.ID, cid)
	}
	if c.Name != "Ada" {
		t.Errorf("name = %q, want %q", c.Name, "Ada")
	}
	if c.Email != "ada@x.io" {
		t.Errorf("email = %q, want %q", c.Email, "ada@x.io")
	}
	if c.CreatedAt == "" {
		t.Error("expected non-empty created_at")
	}
}
