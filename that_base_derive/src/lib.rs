use proc_macro::TokenStream;
use quote::{__private::Span, quote, ToTokens};
use syn::{
    parse_macro_input, punctuated::Punctuated, spanned::Spanned, Data, DataStruct, DeriveInput,
    Field, Fields, Ident, Index, Path, Type, Variant,
};

#[proc_macro_derive(Datum)]
pub fn datum(input: TokenStream) -> TokenStream {
    let DeriveInput {
        ident, generics, ..
    } = parse_macro_input!(input as DeriveInput);
    let (impl_generics, type_generics, where_clauses) = generics.split_for_impl();
    let datum = path(ident.span(), ["that_bass", "Datum"]);
    quote!(
        #[automatically_derived]
        impl #impl_generics #datum for #ident #type_generics #where_clauses { }
    )
    .into()
}

#[proc_macro_derive(Template)]
pub fn template(input: TokenStream) -> TokenStream {
    let DeriveInput {
        ident,
        generics,
        data: Data::Struct(DataStruct { fields, .. }),
        ..
    } = parse_macro_input!(input as DeriveInput) else {
        return quote!(compile_error!("Enumeration and union types are not supported for this derive.");).into();
    };
    let (impl_generics, type_generics, where_clauses) = generics.split_for_impl();
    let template_path = path(ident.span(), ["that_bass", "template", "Template"]);
    let error_path = path(ident.span(), ["that_bass", "Error"]);
    let declare_path = path(ident.span(), ["that_bass", "template", "DeclareContext"]);
    let initialize_path = path(ident.span(), ["that_bass", "template", "InitializeContext"]);
    let apply_path = path(ident.span(), ["that_bass", "template", "ApplyContext"]);
    let (deconstruct, names, types) = deconstruct_fields(&fields);
    let applies = names.iter().enumerate().map(|(i, name)| {
        let index = Index::from(i);
        quote!(#template_path::apply(#name, &_state.#index, _context.own()))
    });
    quote!(
        #[automatically_derived]
        unsafe impl #impl_generics #template_path for #ident #type_generics #where_clauses {
            type State = (#(<#types as #template_path>::State,)*);
            fn declare(mut _context: #declare_path) -> Result<(), #error_path> {
                #(<#types as #template_path>::declare(_context.own())?;)*
                Ok(())
            }
            fn initialize(mut _context: #initialize_path) -> Result<Self::State, #error_path> {
                Ok((#(<#types as #template_path>::initialize(_context.own())?,)*))
            }
            #[inline]
            unsafe fn apply(self, _state: &Self::State, mut _context: #apply_path) {
                let #ident #deconstruct = self;
                #(#applies;)*
            }
        }
    )
    .into()
}

#[proc_macro_derive(Filter)]
pub fn filter(input: TokenStream) -> TokenStream {
    let DeriveInput {
        ident,
        generics,
        data,
        ..
    } = parse_macro_input!(input as DeriveInput);
    let (impl_generics, type_generics, where_clauses) = generics.split_for_impl();
    let filter_path = path(ident.span(), ["that_bass", "filter", "Filter"]);
    let any_path = path(ident.span(), ["that_bass", "filter", "Any"]);
    let table_path = path(ident.span(), ["that_bass", "table", "Table"]);
    let database_path = path(ident.span(), ["that_bass", "Database"]);
    match data {
        Data::Struct(DataStruct { fields, .. }) => {
            let (deconstruct, names, _) = deconstruct_fields(&fields);
            quote!(
                #[automatically_derived]
                impl #impl_generics #filter_path for #ident #type_generics #where_clauses {
                    fn filter(&self, _table: &#table_path, _database: &#database_path) -> bool {
                        let #ident #deconstruct = self;
                        true #(&& #filter_path::filter(#names, _table, _database))*
                    }
                }

                #[automatically_derived]
                impl #impl_generics #filter_path for #any_path<#ident #type_generics> #where_clauses {
                    fn filter(&self,_table: &#table_path, _database: &#database_path) -> bool {
                        let #ident #deconstruct = self.inner();
                        false #(|| #filter_path::filter(#names, _table, _database))*
                    }
                }
            )
        }
        Data::Enum(enumeration) if enumeration.variants.len() == 0 => quote!(compile_error!("Empty enumeration types are not supported for this derive.");),
        Data::Enum(enumeration) => {
            let (all, any): (Vec<_>, Vec<_>) = enumeration
                .variants
                .into_iter()
                .map(|Variant { ident:name, fields, .. }| {
                    let (deconstruct, names, _) = deconstruct_fields(&fields);
                    (
                        quote!(#ident::#name #deconstruct => true #(&& #filter_path::filter(#names, _table, _database))*),
                        quote!(#ident::#name #deconstruct => false #(|| #filter_path::filter(#names, _table, _database))*),
                    )
                })
                .unzip();
            quote!(
                #[automatically_derived]
                impl #impl_generics #filter_path for #ident #type_generics #where_clauses {
                    fn filter(&self, _table: &#table_path, _database: &#database_path) -> bool {
                        match self { #(#all,)* }
                    }
                }

                #[automatically_derived]
                impl #impl_generics #filter_path for #any_path<#ident #type_generics> #where_clauses {
                    fn filter(&self,_table: &#table_path, _database: &#database_path) -> bool {
                        match self.inner() { #(#any,)* }
                    }
                }
            )
        }
        Data::Union(_) => quote!(compile_error!("Union types are not supported for this derive.");),
    }
    .into()
}

#[proc_macro_derive(Row)]
pub fn row(_: TokenStream) -> TokenStream {
    // let DeriveInput {
    //     ident, generics, ..
    // } = parse_macro_input!(input as DeriveInput);
    quote!().into()
}

fn path<'a>(span: Span, segments: impl IntoIterator<Item = &'a str>) -> Path {
    let mut separated = Punctuated::new();
    for segment in segments {
        separated.push(Ident::new(segment, span).into());
    }
    Path {
        segments: separated,
        leading_colon: None,
    }
}

fn deconstruct_fields(fields: &Fields) -> (impl ToTokens, Vec<Ident>, Vec<Type>) {
    match fields {
        Fields::Named(fields) => {
            let (names, types): (Vec<_>, Vec<_>) = fields
                .named
                .iter()
                .filter_map(|Field { ident, ty, .. }| Some((ident.clone()?, ty.clone())))
                .unzip();
            (quote!({ #(#names,)* }), names, types)
        }
        Fields::Unnamed(fields) => {
            let (names, types): (Vec<_>, Vec<_>) = fields
                .unnamed
                .iter()
                .enumerate()
                .map(|(i, field)| {
                    (
                        Ident::new(format!("_{}", i).as_str(), field.span()),
                        field.ty.clone(),
                    )
                })
                .unzip();
            (quote!((#(#names,)*)), names, types)
        }
        Fields::Unit => (quote!(), vec![], vec![]),
    }
}
