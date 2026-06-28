package store

import (
	"fmt"

	lbug "github.com/LadybugDB/go-ladybug"
	"ladycrm/internal/id"
)

// Company represents a row in the Company node table.
type Company struct {
	ID        string
	Name      string
	Domain    string
	Industry  string
	CreatedAt string
}

// AddCompany inserts a new Company node and returns its id.
func AddCompany(conn *lbug.Connection, name string) (string, error) {
	cid := id.New()

	stmt, err := conn.Prepare("CREATE (:Company {id:$id, name:$name})")
	if err != nil {
		return "", fmt.Errorf("prepare add company: %w", err)
	}
	defer stmt.Close()

	_, err = conn.Execute(stmt, map[string]any{
		"id":   cid,
		"name": name,
	})
	if err != nil {
		return "", fmt.Errorf("execute add company: %w", err)
	}
	return cid, nil
}

// ListCompanies returns all companies in the database.
func ListCompanies(conn *lbug.Connection) ([]Company, error) {
	query := "MATCH (c:Company) RETURN c.id, c.name, c.domain, c.industry, c.created_at ORDER BY c.created_at"

	stmt, err := conn.Prepare(query)
	if err != nil {
		return nil, fmt.Errorf("prepare list companies: %w", err)
	}
	defer stmt.Close()

	res, err := conn.Execute(stmt, nil)
	if err != nil {
		return nil, fmt.Errorf("execute list companies: %w", err)
	}
	defer res.Close()

	var companies []Company
	for res.HasNext() {
		tup, err := res.Next()
		if err != nil {
			return nil, fmt.Errorf("next company: %w", err)
		}
		vals, err := tup.GetAsSlice()
		tup.Close()
		if err != nil {
			return nil, fmt.Errorf("get company slice: %w", err)
		}
		companies = append(companies, Company{
			ID:        toString(vals[0]),
			Name:      toString(vals[1]),
			Domain:    toString(vals[2]),
			Industry:  toString(vals[3]),
			CreatedAt: toString(vals[4]),
		})
	}
	return companies, nil
}

// LinkWorksAt matches an existing Contact and Company node and creates a
// WorksAt relationship between them.
func LinkWorksAt(conn *lbug.Connection, contactID, companyID, role string) error {
	stmt, err := conn.Prepare(
		"MATCH (c:Contact {id:$cid}), (co:Company {id:$coID}) CREATE (c)-[:WorksAt {role:$role}]->(co)",
	)
	if err != nil {
		return fmt.Errorf("prepare link works-at: %w", err)
	}
	defer stmt.Close()

	_, err = conn.Execute(stmt, map[string]any{
		"cid":  contactID,
		"coID": companyID,
		"role": role,
	})
	if err != nil {
		return fmt.Errorf("execute link works-at: %w", err)
	}
	return nil
}
