// Package store provides the persistence layer for LadyCRM against LadybugDB.
package store

import (
	"fmt"

	lbug "github.com/LadybugDB/go-ladybug"
	"ladycrm/internal/id"
)

// Contact represents a row in the Contact node table.
type Contact struct {
	ID        string
	Name      string
	Email     string
	Phone     string
	Title     string
	CreatedAt string
}

// AddContact inserts a new Contact node and returns its id.
func AddContact(conn *lbug.Connection, name, email string) (string, error) {
	cid := id.New()
	createdAt := id.Now()

	stmt, err := conn.Prepare("CREATE (:Contact {id:$id, name:$name, email:$email, created_at:$created_at})")
	if err != nil {
		return "", fmt.Errorf("prepare add contact: %w", err)
	}
	defer stmt.Close()

	_, err = conn.Execute(stmt, map[string]any{
		"id":         cid,
		"name":       name,
		"email":      email,
		"created_at": createdAt,
	})
	if err != nil {
		return "", fmt.Errorf("execute add contact: %w", err)
	}
	return cid, nil
}

// ListContacts returns all contacts in the database.
func ListContacts(conn *lbug.Connection) ([]Contact, error) {
	query := "MATCH (c:Contact) RETURN c.id, c.name, c.email, c.phone, c.title, c.created_at ORDER BY c.created_at"

	stmt, err := conn.Prepare(query)
	if err != nil {
		return nil, fmt.Errorf("prepare list contacts: %w", err)
	}
	defer stmt.Close()

	res, err := conn.Execute(stmt, nil)
	if err != nil {
		return nil, fmt.Errorf("execute list contacts: %w", err)
	}
	defer res.Close()

	var contacts []Contact
	for res.HasNext() {
		tup, err := res.Next()
		if err != nil {
			return nil, fmt.Errorf("next contact: %w", err)
		}
		vals, err := tup.GetAsSlice()
		tup.Close()
		if err != nil {
			return nil, fmt.Errorf("get contact slice: %w", err)
		}
		contacts = append(contacts, Contact{
			ID:        toString(vals[0]),
			Name:      toString(vals[1]),
			Email:     toString(vals[2]),
			Phone:     toString(vals[3]),
			Title:     toString(vals[4]),
			CreatedAt: toString(vals[5]),
		})
	}
	return contacts, nil
}

func toString(v any) string {
	if v == nil {
		return ""
	}
	return fmt.Sprintf("%v", v)
}
