// Command crm is the LadyCRM command-line interface. This skeleton builds green
// and proves the LadybugDB linkage (`crm selftest`). Extend the dispatch switch
// with the subcommands described in SPEC.md.
package main

import (
	"flag"
	"fmt"
	"os"

	lbug "github.com/LadybugDB/go-ladybug"
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
	fmt.Fprintln(os.Stderr, "  (build the full surface from SPEC.md)")
}
