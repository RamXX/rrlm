# LadyCRM build. LadybugDB is linked from the system library via the
# `system_ladybug` build tag. Override LBUG_PREFIX if your install differs.
LBUG_PREFIX ?= /opt/homebrew
export CGO_CFLAGS := -I$(LBUG_PREFIX)/include
export CGO_LDFLAGS := -L$(LBUG_PREFIX)/lib -llbug -Wl,-rpath,$(LBUG_PREFIX)/lib
BUILD_TAGS := system_ladybug

.PHONY: build test run clean
build:
	go build -tags $(BUILD_TAGS) -o bin/crm ./cmd/crm

test:
	go test -tags $(BUILD_TAGS) ./...

run: build
	./bin/crm $(ARGS)

clean:
	rm -rf bin
