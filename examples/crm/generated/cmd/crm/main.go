// Command crm is the LadyCRM command-line interface backed by LadybugDB.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"

	lbug "github.com/LadybugDB/go-ladybug"
	"ladycrm/internal/rlm"
	"ladycrm/internal/store"
)

func main() {
	db := flag.String("db", defaultDB(), "path to the LadyCRM database")
	flag.Parse()
	args := flag.Args()
	if len(args) == 0 {
		usage()
		os.Exit(2)
	}
	switch args[0] {
	case "version":
		fmt.Println("ladycrm 0.0.1")
	case "selftest":
		if err := selftest(); err != nil {
			fmt.Fprintln(os.Stderr, "selftest failed:", err)
			os.Exit(1)
		}
		fmt.Println("selftest OK")
	case "init":
		if err := cmdInit(*db); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		fmt.Println("OK")
	case "contact":
		if len(args) < 2 {
			fmt.Fprintln(os.Stderr, "usage: crm contact <add|list> [flags]")
			os.Exit(2)
		}
		switch args[1] {
		case "add":
			if err := cmdContactAdd(*db, args[2:]); err != nil {
				fmt.Fprintln(os.Stderr, err)
				os.Exit(1)
			}
		case "list":
			if err := cmdContactList(*db); err != nil {
				fmt.Fprintln(os.Stderr, err)
				os.Exit(1)
			}
		default:
			fmt.Fprintf(os.Stderr, "unknown contact subcommand %q\n", args[1])
			os.Exit(2)
		}
	case "company":
		if len(args) < 2 {
			fmt.Fprintln(os.Stderr, "usage: crm company <add> [flags]")
			os.Exit(2)
		}
		switch args[1] {
		case "add":
			if err := cmdCompanyAdd(*db, args[2:]); err != nil {
				fmt.Fprintln(os.Stderr, err)
				os.Exit(1)
			}
		default:
			fmt.Fprintf(os.Stderr, "unknown company subcommand %q\n", args[1])
			os.Exit(2)
		}
	case "link":
		if len(args) < 3 {
			fmt.Fprintln(os.Stderr, "usage: crm link works-at --contact <id> --company <id> --role <role>")
			os.Exit(2)
		}
		if args[1] == "works-at" {
			if err := cmdLinkWorksAt(*db, args[2:]); err != nil {
				fmt.Fprintln(os.Stderr, err)
				os.Exit(1)
			}
			fmt.Println("OK")
		} else {
			fmt.Fprintf(os.Stderr, "unknown link type %q\n", args[1])
			os.Exit(2)
		}
	case "deal":
		if len(args) < 2 {
			fmt.Fprintln(os.Stderr, "usage: crm deal <add|stage|list> [flags]")
			os.Exit(2)
		}
		switch args[1] {
		case "add":
			if err := cmdDealAdd(*db, args[2:]); err != nil {
				fmt.Fprintln(os.Stderr, err)
				os.Exit(1)
			}
		case "stage":
			if err := cmdDealStage(*db, args[2:]); err != nil {
				fmt.Fprintln(os.Stderr, err)
				os.Exit(1)
			}
			fmt.Println("OK")
		case "list":
			if err := cmdDealList(*db); err != nil {
				fmt.Fprintln(os.Stderr, err)
				os.Exit(1)
			}
		default:
			fmt.Fprintf(os.Stderr, "unknown deal subcommand %q\n", args[1])
			os.Exit(2)
		}
	case "interact":
		if err := cmdInteract(*db, args[1:]); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	case "timeline":
		if err := cmdTimeline(*db, args[1:]); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	case "path":
		if err := cmdPath(*db, args[1:]); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	case "report":
		if len(args) < 2 {
			fmt.Fprintln(os.Stderr, "usage: crm report \"<question>\"")
			os.Exit(2)
		}
		if err := cmdReport(*db, strings.Join(args[1:], " ")); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	case "import":
		if len(args) < 2 {
			fmt.Fprintln(os.Stderr, "usage: crm import <csvfile>")
			os.Exit(2)
		}
		if err := cmdImport(*db, args[1]); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	default:
		fmt.Fprintf(os.Stderr, "unknown command %q (db=%s)\n", args[0], *db)
		usage()
		os.Exit(2)
	}
}

// selftest opens an in-memory LadybugDB and runs a trivial query, proving the
// native library is linked and callable.
func selftest() error {
	d, err := lbug.OpenInMemoryDatabase(lbug.DefaultSystemConfig())
	if err != nil {
		return err
	}
	defer d.Close()
	c, err := lbug.OpenConnection(d)
	if err != nil {
		return err
	}
	defer c.Close()
	r, err := c.Query("RETURN 1")
	if err != nil {
		return err
	}
	r.Close()
	return nil
}

func defaultDB() string {
	h, _ := os.UserHomeDir()
	return h + "/.ladycrm/db"
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage: crm [--db PATH] <command> [args]")
	fmt.Fprintln(os.Stderr, "  version | selftest")
	fmt.Fprintln(os.Stderr, "  init")
	fmt.Fprintln(os.Stderr, "  contact add --name X --email Y")
	fmt.Fprintln(os.Stderr, "  contact list")
	fmt.Fprintln(os.Stderr, "  company add --name X")
	fmt.Fprintln(os.Stderr, "  link works-at --contact <id> --company <id> --role <role>")
	fmt.Fprintln(os.Stderr, "  deal add --name X --amount N --stage Y --company <id>")
	fmt.Fprintln(os.Stderr, "  deal stage <id> <newstage>")
	fmt.Fprintln(os.Stderr, "  deal list")
	fmt.Fprintln(os.Stderr, "  interact --contact <id> --channel X --summary Y")
	fmt.Fprintln(os.Stderr, "  timeline <contactID>")
	fmt.Fprintln(os.Stderr, "  path --from <contactID> --to <companyID>")
	fmt.Fprintln(os.Stderr, "  report \"<question>\"")
	fmt.Fprintln(os.Stderr, "  import <csvfile>")
}
var schemaDDL = []string{
	`CREATE NODE TABLE IF NOT EXISTS Contact(id STRING, name STRING, email STRING, phone STRING, title STRING, created_at TIMESTAMP, PRIMARY KEY(id))`,
	`CREATE NODE TABLE IF NOT EXISTS Company(id STRING, name STRING, domain STRING, industry STRING, created_at TIMESTAMP, PRIMARY KEY(id))`,
	`CREATE NODE TABLE IF NOT EXISTS Deal(id STRING, name STRING, amount DOUBLE, stage STRING, created_at TIMESTAMP, PRIMARY KEY(id))`,
	`CREATE NODE TABLE IF NOT EXISTS Interaction(id STRING, at TIMESTAMP, channel STRING, summary STRING, PRIMARY KEY(id))`,
	`CREATE REL TABLE IF NOT EXISTS WorksAt(FROM Contact TO Company, role STRING, from_date TIMESTAMP, to_date TIMESTAMP)`,
	`CREATE REL TABLE IF NOT EXISTS DealFor(FROM Deal TO Company)`,
	`CREATE REL TABLE IF NOT EXISTS Had(FROM Contact TO Interaction)`,
}

// cmdInit opens the on-disk database and runs schema DDL idempotently.
func cmdInit(path string) error {
	db, err := lbug.OpenDatabase(path, lbug.DefaultSystemConfig())
	if err != nil {
		return fmt.Errorf("open db: %w", err)
	}
	defer db.Close()
	conn, err := lbug.OpenConnection(db)
	if err != nil {
		return fmt.Errorf("open connection: %w", err)
	}
	defer conn.Close()
	for _, ddl := range schemaDDL {
		_, err := conn.Query(ddl)
		if err != nil {
			return fmt.Errorf("ddl %q: %w", ddl, err)
		}
	}
	return nil
}

// openDB opens an on-disk database and returns an open connection.
func openDB(path string) (*lbug.Database, *lbug.Connection, error) {
	db, err := lbug.OpenDatabase(path, lbug.DefaultSystemConfig())
	if err != nil {
		return nil, nil, fmt.Errorf("open db: %w", err)
	}
	conn, err := lbug.OpenConnection(db)
	if err != nil {
		db.Close()
		return nil, nil, fmt.Errorf("open connection: %w", err)
	}
	return db, conn, nil
}

// cmdContactAdd handles "crm contact add --name X --email Y".
func cmdContactAdd(path string, argv []string) error {
	fs := flag.NewFlagSet("contact add", flag.ExitOnError)
	name := fs.String("name", "", "contact name")
	email := fs.String("email", "", "contact email")
	fs.Parse(argv)
	if *name == "" || *email == "" {
		return fmt.Errorf("--name and --email are required")
	}
	_, conn, err := openDB(path)
	if err != nil {
		return err
	}
	defer conn.Close()
	id, err := store.AddContact(conn, *name, *email)
	if err != nil {
		return err
	}
	fmt.Println(id)
	return nil
}

// cmdContactList handles "crm contact list".
func cmdContactList(path string) error {
	_, conn, err := openDB(path)
	if err != nil {
		return err
	}
	defer conn.Close()
	contacts, err := store.ListContacts(conn)
	if err != nil {
		return err
	}
	if len(contacts) == 0 {
		fmt.Println("(no contacts)")
		return nil
	}
	for _, c := range contacts {
		fmt.Printf("%s\t%s\t%s\n", c.ID, c.Name, c.Email)
	}
	return nil
}

// cmdCompanyAdd handles "crm company add --name X".
func cmdCompanyAdd(path string, argv []string) error {
	fs := flag.NewFlagSet("company add", flag.ExitOnError)
	name := fs.String("name", "", "company name")
	fs.Parse(argv)
	if *name == "" {
		return fmt.Errorf("--name is required")
	}
	_, conn, err := openDB(path)
	if err != nil {
		return err
	}
	defer conn.Close()
	id, err := store.AddCompany(conn, *name)
	if err != nil {
		return err
	}
	fmt.Println(id)
	return nil
}

// cmdLinkWorksAt handles "crm link works-at --contact <id> --company <id> --role <role>".
func cmdLinkWorksAt(path string, argv []string) error {
	fs := flag.NewFlagSet("link works-at", flag.ExitOnError)
	contactID := fs.String("contact", "", "contact id")
	companyID := fs.String("company", "", "company id")
	role := fs.String("role", "", "role title")
	fs.Parse(argv)
	if *contactID == "" || *companyID == "" || *role == "" {
		return fmt.Errorf("--contact, --company, and --role are required")
	}
	_, conn, err := openDB(path)
	if err != nil {
		return err
	}
	defer conn.Close()
	return store.LinkWorksAt(conn, *contactID, *companyID, *role)
}

// cmdDealAdd handles "crm deal add --name X --amount N --stage Y --company <id>".
func cmdDealAdd(path string, argv []string) error {
	fs := flag.NewFlagSet("deal add", flag.ExitOnError)
	name := fs.String("name", "", "deal name")
	amount := fs.Float64("amount", 0, "deal amount")
	stage := fs.String("stage", "", "deal stage")
	companyID := fs.String("company", "", "company id")
	fs.Parse(argv)
	if *name == "" || *companyID == "" || *stage == "" {
		return fmt.Errorf("--name, --stage, and --company are required")
	}
	_, conn, err := openDB(path)
	if err != nil {
		return err
	}
	defer conn.Close()
	id, err := store.AddDeal(conn, *name, *amount, *stage, *companyID)
	if err != nil {
		return err
	}
	fmt.Println(id)
	return nil
}

// cmdDealStage handles "crm deal stage <id> <newstage>".
func cmdDealStage(path string, argv []string) error {
	if len(argv) < 2 {
		return fmt.Errorf("usage: crm deal stage <id> <newstage>")
	}
	_, conn, err := openDB(path)
	if err != nil {
		return err
	}
	defer conn.Close()
	return store.SetStage(conn, argv[0], argv[1])
}

// cmdDealList handles "crm deal list".
func cmdDealList(path string) error {
	_, conn, err := openDB(path)
	if err != nil {
		return err
	}
	defer conn.Close()
	deals, err := store.ListDeals(conn)
	if err != nil {
		return err
	}
	if len(deals) == 0 {
		fmt.Println("(no deals)")
		return nil
	}
	for _, d := range deals {
		fmt.Printf("%s\t%s\t%.2f\t%s\t%s\n", d.ID, d.Name, d.Amount, d.Stage, d.CreatedAt)
	}
	return nil
}

// cmdInteract handles "crm interact --contact <id> --channel X --summary Y".
func cmdInteract(path string, argv []string) error {
	fs := flag.NewFlagSet("interact", flag.ExitOnError)
	contactID := fs.String("contact", "", "contact id")
	channel := fs.String("channel", "", "channel (email, phone, meeting, ...)")
	summary := fs.String("summary", "", "interaction summary")
	fs.Parse(argv)
	if *contactID == "" || *channel == "" || *summary == "" {
		return fmt.Errorf("--contact, --channel, and --summary are required")
	}
	_, conn, err := openDB(path)
	if err != nil {
		return err
	}
	defer conn.Close()
	id, err := store.AddInteraction(conn, *contactID, *channel, *summary)
	if err != nil {
		return err
	}
	fmt.Println(id)
	return nil
}

// cmdTimeline handles "crm timeline <contactID>".
func cmdTimeline(path string, argv []string) error {
	if len(argv) < 1 {
		return fmt.Errorf("usage: crm timeline <contactID>")
	}
	_, conn, err := openDB(path)
	if err != nil {
		return err
	}
	defer conn.Close()
	interactions, err := store.Timeline(conn, argv[0])
	if err != nil {
		return err
	}
	if len(interactions) == 0 {
		fmt.Println("(no interactions)")
		return nil
	}
	for _, i := range interactions {
		fmt.Printf("%s\t%s\t%s\t%s\n", i.At, i.Channel, i.ID, i.Summary)
	}
	return nil
}

// cmdPath handles "crm path --from <contactID> --to <companyID>".
func cmdPath(path string, argv []string) error {
	fs := flag.NewFlagSet("path", flag.ExitOnError)
	from := fs.String("from", "", "contact id")
	to := fs.String("to", "", "company id")
	fs.Parse(argv)
	if *from == "" || *to == "" {
		return fmt.Errorf("--from and --to are required")
	}
	_, conn, err := openDB(path)
	if err != nil {
		return err
	}
	defer conn.Close()
	result, err := store.Path(conn, *from, *to)
	if err != nil {
		return err
	}
	fmt.Println(result)
	return nil
}

// cmdReport gathers a text dump of the graph and asks rlm.Solve to answer the question.
func cmdReport(path string, question string) error {
	_, conn, err := openDB(path)
	if err != nil {
		return err
	}
	defer conn.Close()

	// Gather graph dump.
	var sb strings.Builder

	// Contacts
	contacts, err := store.ListContacts(conn)
	if err != nil {
		return fmt.Errorf("list contacts: %w", err)
	}
	for _, c := range contacts {
		sb.WriteString(fmt.Sprintf("contact %s name=%s email=%s\n", c.ID, c.Name, c.Email))
	}

	// Companies
	companies, err := store.ListCompanies(conn)
	if err != nil {
		return fmt.Errorf("list companies: %w", err)
	}
	for _, co := range companies {
		sb.WriteString(fmt.Sprintf("company %s name=%s\n", co.ID, co.Name))
	}

	// Deals
	deals, err := store.ListDeals(conn)
	if err != nil {
		return fmt.Errorf("list deals: %w", err)
	}
	for _, d := range deals {
		sb.WriteString(fmt.Sprintf("deal %s name=%s amount=%.2f stage=%s\n", d.ID, d.Name, d.Amount, d.Stage))
	}

	// Interactions (all)
	interactions, err := store.ListAllInteractions(conn)
	if err != nil {
		return fmt.Errorf("list interactions: %w", err)
	}
	for _, i := range interactions {
		sb.WriteString(fmt.Sprintf("interaction %s channel=%s summary=%s\n", i.ID, i.Channel, i.Summary))
	}

	answer, err := rlm.Solve(question, sb.String())
	if err != nil {
		return fmt.Errorf("rlm solve: %w", err)
	}
	fmt.Println(answer)
	return nil
}

// cmdImport reads a CSV file and uses rlm.Solve to clean/deduplicate contacts.
func cmdImport(path string, csvFile string) error {
	data, err := os.ReadFile(csvFile)
	if err != nil {
		return fmt.Errorf("read csv: %w", err)
	}

	instruction := "Return a JSON array of cleaned, de-duplicated contacts with name and email. Parse the CSV above, normalize names and emails, remove duplicates (case-insensitive email match), and return a JSON array of objects with 'name' and 'email' keys."
	answer, err := rlm.Solve(instruction, string(data))
	if err != nil {
		return fmt.Errorf("rlm solve: %w", err)
	}

	// Parse JSON array of {name, email}
	type csvContact struct {
		Name  string `json:"name"`
		Email string `json:"email"`
	}
	var contacts []csvContact
	if err := json.Unmarshal([]byte(answer), &contacts); err != nil {
		return fmt.Errorf("parse rlm json: %w (answer=%s)", err, answer)
	}

	_, conn, err := openDB(path)
	if err != nil {
		return err
	}
	defer conn.Close()

	for _, c := range contacts {
		if c.Name == "" || c.Email == "" {
			continue
		}
		id, err := store.AddContact(conn, c.Name, c.Email)
		if err != nil {
			fmt.Fprintf(os.Stderr, "skip contact %s: %v\n", c.Name, err)
			continue
		}
		fmt.Printf("imported %s -> %s\n", c.Name, id)
	}
	return nil
}

// graphDump returns a text dump of all contacts, companies, deals, and interactions.
func graphDump(conn *lbug.Connection) (string, error) {
	var sb strings.Builder

	contacts, err := store.ListContacts(conn)
	if err != nil {
		return "", err
	}
	for _, c := range contacts {
		sb.WriteString(fmt.Sprintf("contact %s name=%s email=%s\n", c.ID, c.Name, c.Email))
	}

	companies, err := store.ListCompanies(conn)
	if err != nil {
		return "", err
	}
	for _, co := range companies {
		sb.WriteString(fmt.Sprintf("company %s name=%s\n", co.ID, co.Name))
	}

	deals, err := store.ListDeals(conn)
	if err != nil {
		return "", err
	}
	for _, d := range deals {
		sb.WriteString(fmt.Sprintf("deal %s name=%s amount=%.2f stage=%s\n", d.ID, d.Name, d.Amount, d.Stage))
	}

	interactions, err := store.ListAllInteractions(conn)
	if err != nil {
		return "", err
	}
	for _, i := range interactions {
		sb.WriteString(fmt.Sprintf("interaction %s channel=%s summary=%s\n", i.ID, i.Channel, i.Summary))
	}

	return sb.String(), nil
}
